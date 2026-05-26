"""Disk persistence for :class:`~src.state.BotState`.

The state file stores everything the bot needs to resume after a hard
restart: which positions it owns, the daily-loss baseline, per-instrument
cooldowns, and the running cycle counter. Cooldowns are stored as
absolute Unix timestamps and re-projected onto the current process'
``time.monotonic()`` clock when loaded, so a 60-min cooldown that was
30 min in still has 30 min to go after a restart.

We deliberately keep the format human-readable JSON and tolerate
missing or stale files — a corrupted state file should never prevent
the bot from starting.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .state import BotState


_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PersistenceMeta:
    """Static metadata pinned to the saved file."""

    saved_at_unix: float
    paused: bool
    schema_version: int = _SCHEMA_VERSION


def _state_to_dict(
    state: BotState,
    *,
    paused: bool,
    autotune_payload: dict | None = None,
    dynamic_stops_payload: dict | None = None,
) -> dict:
    now_mono = time.monotonic()
    now_wall = time.time()
    last_action_wall: dict[str, float] = {}
    for inst_id, mono_ts in state.last_action_per_instrument.items():
        elapsed = max(0.0, now_mono - mono_ts)
        last_action_wall[str(inst_id)] = now_wall - elapsed
    out: dict = {
        "schema_version": _SCHEMA_VERSION,
        "saved_at_unix": now_wall,
        "paused": bool(paused),
        "started_at": state.started_at,
        "session_baseline_equity": state.session_baseline_equity,
        "bot_owned_positions": sorted(int(p) for p in state.bot_owned_positions),
        "last_action_per_instrument_wall": last_action_wall,
        "cycle_count": int(state.cycle_count),
        "halted_today": bool(state.halted_today),
        "halted_day": state.halted_day,
        "bot_actions_today": int(state.bot_actions_today),
        "baseline_day": state.baseline_day,
    }
    if autotune_payload is not None:
        out["autotune"] = autotune_payload
    if dynamic_stops_payload is not None:
        out["dynamic_stops"] = dynamic_stops_payload
    return out


def _state_from_dict(data: dict) -> tuple[BotState, PersistenceMeta]:
    """Rehydrate a BotState; safe against missing keys."""
    state = BotState()
    state.started_at = float(data.get("started_at") or time.time())
    sbe = data.get("session_baseline_equity")
    state.session_baseline_equity = float(sbe) if sbe is not None else None
    state.bot_owned_positions = {int(p) for p in (data.get("bot_owned_positions") or [])}
    state.cycle_count = int(data.get("cycle_count") or 0)
    state.halted_today = bool(data.get("halted_today") or False)
    state.halted_day = data.get("halted_day") or None
    state.bot_actions_today = int(data.get("bot_actions_today") or 0)
    state.baseline_day = data.get("baseline_day") or None

    now_mono = time.monotonic()
    now_wall = time.time()
    last_action_wall = data.get("last_action_per_instrument_wall") or {}
    for raw_id, wall_ts in last_action_wall.items():
        try:
            inst_id = int(raw_id)
            elapsed = max(0.0, now_wall - float(wall_ts))
        except (TypeError, ValueError):
            continue
        state.last_action_per_instrument[inst_id] = now_mono - elapsed

    meta = PersistenceMeta(
        saved_at_unix=float(data.get("saved_at_unix") or 0.0),
        paused=bool(data.get("paused", False)),
        schema_version=int(data.get("schema_version") or _SCHEMA_VERSION),
    )
    return state, meta


class StatePersistence:
    """Atomic JSON save/load for :class:`BotState`.

    Writes go to ``<path>.tmp`` then ``os.replace`` so a crash mid-write
    leaves the previous good file intact.
    """

    def __init__(
        self,
        path: Path,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._path = Path(path)
        self._logger = logger or logging.getLogger("etrader.persistence")

    @property
    def path(self) -> Path:
        return self._path

    def save(
        self,
        state: BotState,
        *,
        paused: bool,
        autotune_payload: dict | None = None,
        dynamic_stops_payload: dict | None = None,
    ) -> None:
        """Persist ``state`` atomically. Logs and swallows any I/O error.

        ``autotune_payload`` is the serialised :class:`AutotuneState`
        snapshot. ``dynamic_stops_payload`` is the per-position SL/TP
        override map from :class:`DynamicStopsStore`. Both are
        optional so legacy call sites that don't know about them
        continue to work — the corresponding block is simply omitted.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            payload = _state_to_dict(
                state,
                paused=paused,
                autotune_payload=autotune_payload,
                dynamic_stops_payload=dynamic_stops_payload,
            )
            tmp.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError as exc:
            self._logger.warning("state save failed: %s", exc)

    def load_dynamic_stops(self) -> dict | None:
        """Return the persisted dynamic-stops block, or None."""
        return self._load_block("dynamic_stops")

    def load_autotune(self) -> dict | None:
        """Return the persisted autotune block from the state file, or None."""
        return self._load_block("autotune")

    def _load_block(self, key: str) -> dict | None:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        block = data.get(key)
        return block if isinstance(block, dict) else None

    def load(self) -> tuple[BotState | None, PersistenceMeta | None]:
        """Return ``(state, meta)`` or ``(None, None)`` if nothing usable."""
        if not self._path.exists():
            return None, None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._logger.warning("state load failed: %s", exc)
            return None, None
        if not isinstance(data, dict):
            return None, None
        try:
            state, meta = _state_from_dict(data)
        except (TypeError, ValueError) as exc:
            self._logger.warning("state file is malformed: %s", exc)
            return None, None
        return state, meta
