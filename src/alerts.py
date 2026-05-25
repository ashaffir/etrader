"""Alert plumbing: alert types, per-chat subscriptions, fan-out queue.

The trading bot emits alerts through :class:`AlertHub`. The hub fans
each alert out to every allowed chat that has subscribed to that type;
unsubscribed chats simply don't see it. The Telegram service drains
pending alerts via the control HTTP API on every poll tick and forwards
them as ordinary chat messages.

Subscription state is per-chat (``data/alert_subscriptions.json``) so
multiple operators can share one bot with different noise levels. New
chats are seeded with the safety-only default (panic close, daily-loss
halt, cycle errors, failed trades) on first contact — opt IN for the
chatty trade-by-trade alerts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable


class AlertType(str, Enum):
    """All alert categories the bot can emit.

    String-valued so JSON persistence and HTTP wire format are obvious;
    keep names stable — they're persisted in
    ``data/alert_subscriptions.json``.
    """

    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    TRADE_FAILED = "trade_failed"
    PANIC_CLOSE = "panic_close"
    DAILY_LOSS_HALT = "daily_loss_halt"
    CYCLE_ERROR = "cycle_error"
    AI_UNAVAILABLE = "ai_unavailable"
    UNIVERSE_CHANGED = "universe_changed"
    # Emitted when the universe refresh produced rejections — i.e. a
    # candidate the news pipeline flagged was filtered out by the
    # activity gate (low ATR, wide spread, ...). Helps operators tune
    # `[universe] min_atr_pct` / `max_spread_pct` and notice when a
    # promising news story isn't tradeable on eToro.
    UNIVERSE_REJECTED = "universe_rejected"
    BOT_PAUSED_RESUMED = "bot_paused_resumed"

    @classmethod
    def all_types(cls) -> list["AlertType"]:
        return list(cls)

    @classmethod
    def from_value(cls, raw: str) -> "AlertType | None":
        try:
            return cls(raw)
        except ValueError:
            return None


# The "safety-only" default: critical events most operators always want
# to know about, without the trade-by-trade noise.
_SAFETY_ONLY_DEFAULT = frozenset({
    AlertType.PANIC_CLOSE,
    AlertType.DAILY_LOSS_HALT,
    AlertType.CYCLE_ERROR,
    AlertType.TRADE_FAILED,
})


def safety_only_default() -> set[AlertType]:
    """Default subscription set for a brand-new chat."""
    return set(_SAFETY_ONLY_DEFAULT)


@dataclass(frozen=True)
class Alert:
    type: AlertType
    timestamp: str  # ISO-8601 UTC seconds resolution
    title: str
    body: str

    def to_telegram(self) -> str:
        if self.body:
            return f"[{self.title}]\n{self.body}"
        return f"[{self.title}]"

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "title": self.title,
            "body": self.body,
        }


# ---------------------------------------------------------------------------
# Subscription store — per-chat, persisted
# ---------------------------------------------------------------------------

class AlertSubscriptions:
    """Per-chat ``set[AlertType]``, persisted as JSON.

    Reads are O(1) per chat; writes flush atomically (write-and-rename)
    so a crash mid-save can't corrupt the store. Unknown chat IDs auto-
    populate with the configured default on first read.
    """

    def __init__(
        self,
        path: Path,
        *,
        default_set: Iterable[AlertType] | None = None,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._path = Path(path)
        self._default = set(default_set) if default_set is not None else safety_only_default()
        self._lock = threading.Lock()
        self._by_chat: dict[int, set[AlertType]] = {}
        self._logger = logger or logging.getLogger("etrader.alerts.subs")
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._logger.warning("alert subs load failed: %s", exc)
            return
        if not isinstance(data, dict):
            return
        for chat_str, types_list in data.items():
            try:
                chat_id = int(chat_str)
            except (TypeError, ValueError):
                continue
            valid: set[AlertType] = set()
            for raw in types_list or []:
                resolved = AlertType.from_value(str(raw))
                if resolved is not None:
                    valid.add(resolved)
            self._by_chat[chat_id] = valid

    def _save_locked(self) -> None:
        out = {
            str(chat): sorted(t.value for t in types)
            for chat, types in self._by_chat.items()
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            self._logger.warning("alert subs save failed: %s", exc)

    def enabled_for(self, chat_id: int) -> set[AlertType]:
        chat_id = int(chat_id)
        with self._lock:
            if chat_id not in self._by_chat:
                self._by_chat[chat_id] = set(self._default)
                self._save_locked()
            return set(self._by_chat[chat_id])

    def set_enabled(
        self, chat_id: int, alert_type: AlertType, enabled: bool,
    ) -> set[AlertType]:
        chat_id = int(chat_id)
        with self._lock:
            current = self._by_chat.setdefault(chat_id, set(self._default))
            if enabled:
                current.add(alert_type)
            else:
                current.discard(alert_type)
            self._save_locked()
            return set(current)

    def toggle(self, chat_id: int, alert_type: AlertType) -> tuple[bool, set[AlertType]]:
        """Flip a single alert. Returns ``(new_state, full_enabled_set)``."""
        chat_id = int(chat_id)
        with self._lock:
            current = self._by_chat.setdefault(chat_id, set(self._default))
            new_state = alert_type not in current
            if new_state:
                current.add(alert_type)
            else:
                current.discard(alert_type)
            self._save_locked()
            return new_state, set(current)

    def reset_to_default(self, chat_id: int) -> set[AlertType]:
        chat_id = int(chat_id)
        with self._lock:
            self._by_chat[chat_id] = set(self._default)
            self._save_locked()
            return set(self._by_chat[chat_id])


# ---------------------------------------------------------------------------
# Hub — emit + drain queue per chat
# ---------------------------------------------------------------------------

class AlertHub:
    """Thread-safe alert fan-out + per-chat in-memory queue.

    The bot's emitters never block: a full queue silently drops the
    oldest alert (``deque(maxlen=...)`` semantics). The Telegram service
    drains via the HTTP API and forwards to chats.

    Allowed chat IDs are taken from ``.env`` (the same list the Telegram
    service uses) so the bot knows where to fan out to without an
    explicit registration call.
    """

    def __init__(
        self,
        *,
        allowed_chat_ids: Iterable[int],
        subscriptions: AlertSubscriptions,
        max_per_chat: int = 200,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._chat_ids: tuple[int, ...] = tuple(int(c) for c in allowed_chat_ids)
        self._subs = subscriptions
        self._lock = threading.Lock()
        self._queues: dict[int, deque[Alert]] = {
            c: deque(maxlen=max_per_chat) for c in self._chat_ids
        }
        self._logger = logger or logging.getLogger("etrader.alerts.hub")

    @property
    def allowed_chat_ids(self) -> tuple[int, ...]:
        return self._chat_ids

    @property
    def subscriptions(self) -> AlertSubscriptions:
        return self._subs

    def emit(
        self,
        alert_type: AlertType,
        *,
        title: str,
        body: str = "",
    ) -> None:
        if not self._chat_ids:
            return
        alert = Alert(
            type=alert_type,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            title=title,
            body=body,
        )
        delivered = 0
        with self._lock:
            for chat_id in self._chat_ids:
                if alert_type in self._subs.enabled_for(chat_id):
                    self._queues[chat_id].append(alert)
                    delivered += 1
        self._logger.debug(
            "[alerts] emit %s -> %d/%d chat(s)",
            alert_type.value, delivered, len(self._chat_ids),
        )

    def drain(self, chat_id: int, *, limit: int = 50) -> list[Alert]:
        """Pop up to ``limit`` queued alerts for one chat (oldest first)."""
        chat_id = int(chat_id)
        limit = max(1, int(limit))
        with self._lock:
            q = self._queues.get(chat_id)
            if not q:
                return []
            out: list[Alert] = []
            while q and len(out) < limit:
                out.append(q.popleft())
            return out

    def queue_depth(self, chat_id: int) -> int:
        with self._lock:
            q = self._queues.get(int(chat_id))
            return len(q) if q else 0

    @staticmethod
    def list_alert_types() -> list[AlertType]:
        return AlertType.all_types()
