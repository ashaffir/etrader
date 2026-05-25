"""In-memory bot state.

Tracks the cooldowns, daily-loss baseline, and which positions were
opened by *this* bot session (so we don't accidentally close positions
the user already had).

State is intentionally non-persistent: a restart resets cooldowns and
the daily-loss baseline. This keeps recovery simple and avoids having
the bot mis-attribute positions if the state file is stale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BotState:
    """Per-process mutable state.

    Attributes
    ----------
    started_at:
        Wallclock seconds at process start.
    session_baseline_equity:
        Equity at session start (filled lazily after the first PnL read).
        Used by :class:`risk.DailyLossKillSwitch` to decide whether to halt.
    last_action_per_instrument:
        ``{instrument_id: monotonic_seconds_at_last_action}`` — used by the
        cooldown guard to skip recently-touched instruments.
    bot_owned_positions:
        Position IDs that this session opened. The executor only ever
        closes positions whose IDs are in here.
    cycle_count:
        Number of completed main-loop cycles (1-indexed in logs).
    halted_today:
        Set when the daily-loss kill switch fires; cleared on next UTC day.
    """

    started_at: float = field(default_factory=time.time)
    session_baseline_equity: float | None = None
    last_action_per_instrument: dict[int, float] = field(default_factory=dict)
    bot_owned_positions: set[int] = field(default_factory=set)
    # Instrument IDs corresponding to currently open, bot-owned positions.
    # Updated by the cycle right after every portfolio reconcile so the
    # universe builder can pin them as `must_include` on the next
    # refresh (we never want to lose sight of a position we own).
    bot_owned_instrument_ids: dict[int, str] = field(default_factory=dict)
    cycle_count: int = 0
    halted_today: bool = False
    halted_day: str | None = None  # YYYY-MM-DD UTC, used to auto-reset
    bot_actions_today: int = 0     # successful BUY/CLOSE counter; resets daily
    baseline_day: str | None = None  # day the equity baseline was captured

    def mark_action(self, instrument_id: int) -> None:
        self.last_action_per_instrument[instrument_id] = time.monotonic()

    def record_bot_action(self) -> None:
        """Increment the daily counter; called by the executor on success."""
        self.bot_actions_today += 1

    def seconds_since_action(self, instrument_id: int) -> float | None:
        ts = self.last_action_per_instrument.get(instrument_id)
        if ts is None:
            return None
        return time.monotonic() - ts

    def add_owned(self, position_id: int) -> None:
        self.bot_owned_positions.add(position_id)

    def remove_owned(self, position_id: int) -> None:
        self.bot_owned_positions.discard(position_id)
