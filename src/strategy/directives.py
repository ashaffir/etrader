"""Persistent operator directives — hybrid structured + free-text overlay.

The operator uses Telegram to attach long-lived rules the bot must
honour every cycle: "no overnight holds", "never buy NVDA", "the
account budget across bot + manual is $X", etc.  These are persisted
to ``bot_state.json`` (like dynamic stops / autotune) so a restart
preserves the operator's intent.

Two channels coexist:

* **Structured fields** — a small enumerated schema the code can
  enforce deterministically (see :class:`Directives`). The Telegram
  ``/directive set <key> <value>`` editor only accepts these keys.
* **Free-text notes** — a single ``notes`` string the LLM consults
  in every cycle prompt for soft directives the schema doesn't
  capture (e.g. *"prefer financial-sector names this week"*). Code
  never enforces notes; the LLM does.

The store is thread-safe (cycle thread + control thread both touch
it) and exposes ``to_persistable`` / ``restore`` so it slots into
:class:`~src.persistence.StatePersistence` the same way the
DynamicStopsStore does.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, fields as dc_fields, replace
from typing import Any


# Cap on free-text notes — Telegram bot UX gets unwieldy past ~500
# chars and the LLM prompt budget is finite. We keep this generous
# but bounded to prevent runaway growth.
NOTES_MAX_CHARS = 500


@dataclass(frozen=True)
class Directives:
    """Immutable snapshot of the operator's persistent directives.

    Every field defaults to a *disabled* state so an absent directive
    set behaves exactly like the pre-directives bot. Booleans default
    to False, numeric caps default to 0 (interpreted as "disabled"),
    list fields default to empty tuples, ``notes`` defaults to "".
    """

    # Close any bot-opened position before its US market closes for
    # the day. Crypto positions are exempt (24/7 markets). Enforced
    # by :mod:`src.strategy.directive_enforcer` injecting CLOSE
    # actions into the cycle when the local UTC time is within the
    # flatten window.
    no_overnight: bool = False

    # Auto-close any bot-opened position held continuously for more
    # than this many minutes. Measured from the position's open time
    # as recorded by :class:`PerformanceTracker`. 0 = disabled.
    hold_ceiling_minutes: int = 0

    # Symbols the bot must never open. Stored uppercase. The risk
    # evaluator refuses BUYs targeting any of these; the cycle layer
    # also drops them from candidate lists before they reach the LLM
    # so the prompt isn't polluted.
    blocked_symbols: tuple[str, ...] = ()

    # Sectors the bot must never open (matched against the
    # fundamentals cache's ``sector`` field — case-insensitive).
    # Enforced at the cycle level (the risk layer doesn't see
    # sector). Empty = no restriction.
    blocked_sectors: tuple[str, ...] = ()

    # Refuse any new BUY that would push the total invested across
    # the WHOLE eToro account (bot + manual + mirror) above this USD
    # value. 0 = disabled. Does NOT close manual positions — the
    # bot is forbidden from touching those.
    max_total_account_invested_usd: float = 0.0

    # Flatten any non-crypto bot-owned position when its next
    # scheduled earnings call is inside this window (hours). Same
    # rationale as ``no_overnight``: earnings gaps are a fundamental
    # gamble the bot can't predict. 0 = disabled. Requires the
    # earnings calendar to be enabled in ``[earnings_calendar]``.
    pre_earnings_close_hours: int = 0

    # Free-text soft directives. Surfaced to the LLM verbatim in
    # every decision prompt. Bounded by :data:`NOTES_MAX_CHARS`.
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "no_overnight": bool(self.no_overnight),
            "hold_ceiling_minutes": int(self.hold_ceiling_minutes),
            "blocked_symbols": list(self.blocked_symbols),
            "blocked_sectors": list(self.blocked_sectors),
            "max_total_account_invested_usd": float(
                self.max_total_account_invested_usd
            ),
            "pre_earnings_close_hours": int(self.pre_earnings_close_hours),
            "notes": str(self.notes or ""),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "Directives":
        if not payload:
            return cls()
        return cls(
            no_overnight=_coerce_bool(payload.get("no_overnight")),
            hold_ceiling_minutes=_coerce_nonneg_int(payload.get("hold_ceiling_minutes")),
            blocked_symbols=_coerce_symbol_tuple(payload.get("blocked_symbols")),
            blocked_sectors=_coerce_label_tuple(payload.get("blocked_sectors")),
            max_total_account_invested_usd=_coerce_nonneg_float(
                payload.get("max_total_account_invested_usd")
            ),
            pre_earnings_close_hours=_coerce_nonneg_int(
                payload.get("pre_earnings_close_hours")
            ),
            notes=_coerce_notes(payload.get("notes")),
        )

    def is_symbol_blocked(self, symbol: str) -> bool:
        sym = (symbol or "").strip().upper()
        return bool(sym) and sym in self.blocked_symbols

    def is_sector_blocked(self, sector: str | None) -> bool:
        if not sector:
            return False
        norm = sector.strip().lower()
        return any(s.strip().lower() == norm for s in self.blocked_sectors)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

# Public list of editable structured keys. The Telegram + HTTP
# editors look at this to validate operator input + render the menu.
STRUCTURED_KEYS: tuple[str, ...] = (
    "no_overnight",
    "hold_ceiling_minutes",
    "blocked_symbols",
    "blocked_sectors",
    "max_total_account_invested_usd",
    "pre_earnings_close_hours",
)

_FIELD_TYPES: dict[str, str] = {
    "no_overnight": "bool",
    "hold_ceiling_minutes": "int",
    "blocked_symbols": "symbol_list",
    "blocked_sectors": "label_list",
    "max_total_account_invested_usd": "float",
    "pre_earnings_close_hours": "int",
}


class DirectiveError(ValueError):
    """Raised when an operator sets an unknown / malformed directive."""


def coerce_value(key: str, raw: Any) -> Any:
    """Validate + coerce a single directive value for ``key``.

    Raises :class:`DirectiveError` on unknown keys or junk input.
    The Telegram and HTTP editors both use this so the rules live
    in exactly one place.
    """
    kind = _FIELD_TYPES.get(key)
    if kind is None:
        raise DirectiveError(
            f"unknown directive {key!r}. Allowed: {', '.join(STRUCTURED_KEYS)}"
        )
    if kind == "bool":
        return _coerce_bool(raw, strict=True)
    if kind == "int":
        return _coerce_nonneg_int(raw, strict=True)
    if kind == "float":
        return _coerce_nonneg_float(raw, strict=True)
    if kind == "symbol_list":
        return _coerce_symbol_tuple(raw, strict=True)
    if kind == "label_list":
        return _coerce_label_tuple(raw, strict=True)
    raise DirectiveError(f"unsupported field kind {kind!r}")  # pragma: no cover


def _coerce_bool(raw: Any, *, strict: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"true", "yes", "on", "1"}:
            return True
        if s in {"false", "no", "off", "0", ""}:
            return False
    if strict:
        raise DirectiveError(f"expected bool-like value, got {raw!r}")
    return False


def _coerce_nonneg_int(raw: Any, *, strict: bool = False) -> int:
    try:
        value = int(float(raw))  # tolerate "60.0"
    except (TypeError, ValueError):
        if strict:
            raise DirectiveError(f"expected non-negative integer, got {raw!r}") from None
        return 0
    if value < 0:
        if strict:
            raise DirectiveError(f"value must be >= 0, got {value}")
        return 0
    return value


def _coerce_nonneg_float(raw: Any, *, strict: bool = False) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        if strict:
            raise DirectiveError(f"expected non-negative number, got {raw!r}") from None
        return 0.0
    if value < 0:
        if strict:
            raise DirectiveError(f"value must be >= 0, got {value}")
        return 0.0
    return value


def _coerce_symbol_tuple(raw: Any, *, strict: bool = False) -> tuple[str, ...]:
    items = _split_list_input(raw)
    cleaned: list[str] = []
    for item in items:
        token = (item or "").strip().upper()
        if not token:
            continue
        # eToro symbols are alphanum + '.' / '-' (e.g. BRK.B). Reject
        # anything else — the LLM doesn't gain anything from blocking
        # malformed strings and they'd cause silent mismatches.
        if not all(c.isalnum() or c in {".", "-", "/"} for c in token):
            if strict:
                raise DirectiveError(f"invalid symbol {item!r}")
            continue
        if token not in cleaned:
            cleaned.append(token)
    return tuple(cleaned)


def _coerce_label_tuple(raw: Any, *, strict: bool = False) -> tuple[str, ...]:
    items = _split_list_input(raw)
    cleaned: list[str] = []
    for item in items:
        token = (item or "").strip()
        if not token:
            continue
        key = token.lower()
        if key not in {c.lower() for c in cleaned}:
            cleaned.append(token)
    return tuple(cleaned)


def _split_list_input(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return [p for p in raw.replace(";", ",").split(",")]
    return [str(raw)]


def _coerce_notes(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if len(text) > NOTES_MAX_CHARS:
        text = text[:NOTES_MAX_CHARS]
    return text


# ---------------------------------------------------------------------------
# Mutable store — what runtime code interacts with
# ---------------------------------------------------------------------------


class DirectivesStore:
    """Thread-safe holder of the live :class:`Directives` snapshot.

    The trading loop reads from it (``current()``) on every cycle;
    Telegram / HTTP handlers write to it. Mutations always replace
    the whole snapshot atomically so readers never see a half-edit.
    """

    def __init__(
        self,
        initial: Directives | None = None,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._directives = initial or Directives()
        self._log = logger or logging.getLogger("etrader.strategy.directives")

    # -- reads ----------------------------------------------------------

    def current(self) -> Directives:
        with self._lock:
            return self._directives

    def to_dict(self) -> dict[str, Any]:
        return self.current().to_dict()

    # -- structured edits ----------------------------------------------

    def set_field(self, key: str, raw: Any) -> tuple[Any, Any]:
        """Update one structured field. Returns ``(previous, current)``.

        Validates + coerces ``raw`` via :func:`coerce_value`. Raises
        :class:`DirectiveError` on unknown keys / malformed values.
        """
        coerced = coerce_value(key, raw)
        with self._lock:
            previous = getattr(self._directives, key)
            self._directives = replace(self._directives, **{key: coerced})
        self._log.info("[directives] %s: %r → %r", key, previous, coerced)
        return previous, coerced

    def clear_field(self, key: str) -> tuple[Any, Any]:
        """Reset a structured field to its default. Returns ``(prev, current)``."""
        if key not in _FIELD_TYPES:
            raise DirectiveError(
                f"unknown directive {key!r}. Allowed: {', '.join(STRUCTURED_KEYS)}"
            )
        default = next(
            f.default for f in dc_fields(Directives) if f.name == key
        )
        with self._lock:
            previous = getattr(self._directives, key)
            self._directives = replace(self._directives, **{key: default})
        self._log.info("[directives] %s: %r → cleared (%r)", key, previous, default)
        return previous, default

    # -- notes ----------------------------------------------------------

    def set_notes(self, text: str) -> tuple[str, str]:
        coerced = _coerce_notes(text)
        with self._lock:
            previous = self._directives.notes
            self._directives = replace(self._directives, notes=coerced)
        self._log.info(
            "[directives] notes: %d → %d chars", len(previous), len(coerced),
        )
        return previous, coerced

    def clear_notes(self) -> str:
        with self._lock:
            previous = self._directives.notes
            self._directives = replace(self._directives, notes="")
        self._log.info("[directives] notes cleared (%d chars)", len(previous))
        return previous

    # -- persistence ----------------------------------------------------

    def to_persistable(self) -> dict[str, Any]:
        return self.current().to_dict()

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        with self._lock:
            self._directives = Directives.from_dict(payload)


__all__ = [
    "Directives",
    "DirectivesStore",
    "DirectiveError",
    "STRUCTURED_KEYS",
    "NOTES_MAX_CHARS",
    "coerce_value",
]
