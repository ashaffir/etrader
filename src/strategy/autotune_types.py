"""Data shapes for the autonomous-tuner overlay.

The autotuner is the "manager LLM" tier: every cycle it receives a
digest of how the bot is doing and may optionally return a tuning
block that edits any field in ``[strategy]`` or ``[tools.spread_max_pct]``.
This module defines the *contracts* between the cycle, the LLM prompt,
and the apply step; the actual state machine lives in
:mod:`src.strategy.autotune_state`.

Why a dedicated tuner?
----------------------
Static thresholds in ``config.toml`` are a bug, not a feature, for a
truly autonomous bot. ``min_signal_strength = 0.40`` requires multiple
fresh triggers firing simultaneously, which is a "perfect setup" gate;
on a calm market it produces zero candidates for hours. The decision
LLM was already seeing the candidate set; we just need to let it also
tune the gate that *produces* the candidate set.

Wire shape (LLM JSON → engine):

.. code-block:: json

    {
      "actions": [ ... ],
      "summary": "...",
      "tuning": {
        "changes": [
          {"section": "strategy", "field": "min_signal_strength",
           "value": 0.30, "rationale": "no candidates for 4h ..."}
        ],
        "reason": "loosen entry; drought detected"
      }
    }

The ``tuning`` key is optional; absence means "no change this cycle".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Sections the LLM is allowed to touch. Anything else in a TuneRequest
# is silently dropped — keeps a hallucinated "section": "guardrails"
# from sneaking in a daily-loss override through the tuner channel.
ALLOWED_SECTIONS: tuple[str, ...] = ("strategy", "tools")


# Per-section whitelists. Listed explicitly so a typo in the LLM
# output (e.g. ``min_signal_strenght``) can't quietly fail by writing
# an unused row into the SQLite store.
STRATEGY_FIELDS: tuple[str, ...] = (
    "sma_short_period", "sma_long_period",
    "ema_fast_period", "ema_slow_period",
    "rsi_period", "rsi_oversold", "rsi_overbought",
    "macd_fast", "macd_slow", "macd_signal",
    "bollinger_period", "bollinger_stddev",
    "donchian_period", "momentum_lookback",
    "min_signal_strength", "min_exit_strength",
    "weight_sma_cross", "weight_ema_cross",
    "weight_rsi", "weight_macd",
    "weight_bollinger", "weight_donchian", "weight_momentum",
)

TOOLS_FIELDS: tuple[str, ...] = (
    "spread_max_pct",
)


# Fields that must be integer-typed once coerced. Everything else is
# coerced to float. We do not bound values per the operator's explicit
# preference (full autonomy); we only protect against type errors so
# downstream code doesn't crash on e.g. ``rsi_period=14.0`` ints.
_INT_FIELDS: frozenset[str] = frozenset({
    "sma_short_period", "sma_long_period",
    "ema_fast_period", "ema_slow_period",
    "rsi_period",
    "macd_fast", "macd_slow", "macd_signal",
    "bollinger_period",
    "donchian_period", "momentum_lookback",
})


def field_kind(section: str, field_name: str) -> str:
    """Return ``"int"``, ``"float"`` or ``"unknown"`` for a (section, field)."""
    if section == "strategy":
        if field_name in _INT_FIELDS:
            return "int"
        if field_name in STRATEGY_FIELDS:
            return "float"
    if section == "tools" and field_name in TOOLS_FIELDS:
        return "float"
    return "unknown"


@dataclass(frozen=True)
class TuneChange:
    """A single ``(section, field, value)`` edit proposed by the LLM."""

    section: str
    field: str
    value: Any
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "field": self.field,
            "value": self.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class TuneRequest:
    """A parsed, coerced batch of changes the LLM emitted this cycle.

    Empty ``changes`` means "no-op"; the engine returns this when the
    LLM did not include a ``tuning`` block at all so callers can use a
    single code path.
    """

    changes: tuple[TuneChange, ...] = ()
    reason: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "changes": [c.to_dict() for c in self.changes],
        }


@dataclass(frozen=True)
class TuneApplied:
    """Result of applying a :class:`TuneRequest` to the live config.

    Captures both the requested value and the previous value so the
    Telegram alert can show a diff and so the rolling autotune log
    can record the actual transition (not just the proposal).
    """

    section: str
    field: str
    previous: Any
    current: Any
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "field": self.field,
            "previous": self.previous,
            "current": self.current,
            "rationale": self.rationale,
        }


@dataclass
class AutotuneEvidence:
    """Per-cycle digest the LLM sees before deciding whether to tune.

    Mutable on purpose so the cycle can fill it incrementally as more
    information becomes available (e.g. the raw_score histogram is
    computed before the LLM call; recent trade P&L is filled in by
    the cycle wrapper).

    Every field is JSON-serializable so the prompt builder can dump
    it verbatim without further transformation.
    """

    cycle_index: int = 0
    tracked_count: int = 0
    candidates_this_cycle: int = 0
    top_raw_score: float | None = None
    raw_score_distribution: dict[str, float] = field(default_factory=dict)
    cycles_since_last_candidate: int = 0
    cycles_since_last_trade: int = 0
    cycles_since_last_fill: int = 0
    last_n_cycles: list[dict[str, Any]] = field(default_factory=list)
    recent_realized_pnl: list[dict[str, Any]] = field(default_factory=list)
    open_position_pnl_total: float = 0.0
    previous_tunings: list[dict[str, Any]] = field(default_factory=list)
    current_thresholds: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    current_spread_max_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render to a plain dict suitable for the LLM prompt JSON body.

        Deliberately omits ``trading_mode``: per the project-wide
        invariant the decision-call LLM must reason identically in
        paper vs live, so the evidence digest does not surface it
        either.
        """
        return {
            "cycle_index": int(self.cycle_index),
            "tracked_count": int(self.tracked_count),
            "candidates_this_cycle": int(self.candidates_this_cycle),
            "top_raw_score": self.top_raw_score,
            "raw_score_distribution": dict(self.raw_score_distribution),
            "drought": {
                "cycles_since_last_candidate": int(self.cycles_since_last_candidate),
                "cycles_since_last_trade": int(self.cycles_since_last_trade),
                "cycles_since_last_fill": int(self.cycles_since_last_fill),
            },
            "last_n_cycles": list(self.last_n_cycles),
            "recent_realized_pnl": list(self.recent_realized_pnl),
            "open_position_pnl_total": float(self.open_position_pnl_total),
            "previous_tunings": list(self.previous_tunings),
            "current_thresholds": dict(self.current_thresholds),
            "current_weights": dict(self.current_weights),
            "current_spread_max_pct": self.current_spread_max_pct,
        }
