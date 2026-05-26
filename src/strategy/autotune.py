"""Autonomous-tuner state machine + apply path.

The cycle owns an :class:`AutotuneState` instance for the lifetime of
the process. Each cycle it:

1. Calls :meth:`AutotuneState.observe_cycle` to fold this cycle's
   raw-score histogram + candidate count into the rolling window.
2. Calls :meth:`AutotuneState.build_evidence` to materialise the
   payload that goes into the LLM prompt.
3. After the LLM returns, calls :meth:`AutotuneState.apply` with the
   parsed :class:`TuneRequest`. The apply step:
   - mutates ``cfg.strategy`` / ``cfg.tools`` in place;
   - writes the new values to the SQLite config store so they survive
     a restart;
   - appends to the rolling autotune log;
   - returns a list of :class:`TuneApplied` records the caller turns
     into a Telegram alert and a log line.

This module deliberately knows nothing about prompts or alerts so it
can be unit-tested without spinning up an LLM client.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from ..config import AppConfig
from ..config_store import ConfigStore
from .autotune_parse import render_tune_diff
from .autotune_types import (
    AutotuneEvidence,
    TuneApplied,
    TuneRequest,
)


# Rolling window for "last N cycles" and "previous tunings". Tuned for
# a 1-minute cycle: 60 entries ≈ 1 hour, plenty for the LLM to see a
# trend without bloating the prompt.
_DEFAULT_WINDOW = 60
_DEFAULT_TUNING_LOG = 30


@dataclass
class _CycleSnapshot:
    """One row of the rolling per-cycle window."""

    cycle_index: int
    timestamp_unix: float
    tracked: int
    candidates: int
    top_raw_score: float | None
    trades_placed: int
    raw_scores: tuple[float, ...]

    def to_summary(self) -> dict[str, Any]:
        return {
            "cycle": int(self.cycle_index),
            "tracked": int(self.tracked),
            "candidates": int(self.candidates),
            "top_raw_score": self.top_raw_score,
            "trades_placed": int(self.trades_placed),
        }


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    """Inclusive percentile for an already-sorted list. Returns ``None`` if empty."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pct = max(0.0, min(100.0, float(pct)))
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _distribution(values: Iterable[float]) -> dict[str, float]:
    """Compact stats: min / p25 / median / p75 / p90 / max + mean."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {}
    vals_sorted = sorted(vals)
    return {
        "n": float(len(vals)),
        "min": float(vals_sorted[0]),
        "p25": float(_percentile(vals_sorted, 25) or vals_sorted[0]),
        "median": float(_percentile(vals_sorted, 50) or vals_sorted[0]),
        "p75": float(_percentile(vals_sorted, 75) or vals_sorted[0]),
        "p90": float(_percentile(vals_sorted, 90) or vals_sorted[0]),
        "max": float(vals_sorted[-1]),
        "mean": float(statistics.fmean(vals)),
    }


class AutotuneState:
    """Process-lifetime rolling state for the autonomous tuner.

    Thread-safety: the cycle holds ``controller.lock`` for the whole
    cycle, so single-thread is sufficient. The store is the only
    cross-thread surface and it has its own lock internally.
    """

    def __init__(
        self,
        *,
        config_store: ConfigStore | None,
        window: int = _DEFAULT_WINDOW,
        tuning_log_size: int = _DEFAULT_TUNING_LOG,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._store = config_store
        self._log = logger or logging.getLogger("etrader.strategy.autotune")
        self._cycles: deque[_CycleSnapshot] = deque(maxlen=int(window))
        self._tunings: deque[dict[str, Any]] = deque(maxlen=int(tuning_log_size))
        self._cycles_since_last_candidate: int = 0
        self._cycles_since_last_trade: int = 0
        self._cycles_since_last_fill: int = 0

    # ------------------------------------------------------------------
    # Cycle ingest
    # ------------------------------------------------------------------

    def observe_cycle(
        self,
        *,
        cycle_index: int,
        tracked_count: int,
        raw_scores: Iterable[float],
        candidates_count: int,
    ) -> None:
        """Open a new cycle snapshot with pre-decision data.

        Call this RIGHT after candidates are built so the LLM sees
        this cycle's raw_score histogram in the evidence digest. The
        ``trades_placed`` field defaults to 0 here and is finalised
        post-execution via :meth:`record_trades_placed`.
        """
        scores = tuple(float(s) for s in raw_scores if s is not None)
        top = max(scores) if scores else None
        snap = _CycleSnapshot(
            cycle_index=int(cycle_index),
            timestamp_unix=time.time(),
            tracked=int(tracked_count),
            candidates=int(candidates_count),
            top_raw_score=top,
            trades_placed=0,
            raw_scores=scores,
        )
        self._cycles.append(snap)
        # Candidate-side drought counter is final at this point.
        if snap.candidates > 0:
            self._cycles_since_last_candidate = 0
        else:
            self._cycles_since_last_candidate += 1
        # Trades drought is provisionally incremented; record_trades_placed
        # rolls it back to 0 if the cycle actually fired any orders.
        self._cycles_since_last_trade += 1
        self._cycles_since_last_fill += 1

    def record_trades_placed(self, *, trades_placed: int) -> None:
        """Finalise this cycle's trade count (called post-execution)."""
        if not self._cycles:
            return
        last = self._cycles[-1]
        # Tuple-of-tuples deque holds frozen-ish snapshots; rebuild the
        # last entry rather than mutating a dataclass field across
        # threads. (The snapshot dataclass is mutable but we keep the
        # mutation localised here.)
        updated = _CycleSnapshot(
            cycle_index=last.cycle_index,
            timestamp_unix=last.timestamp_unix,
            tracked=last.tracked,
            candidates=last.candidates,
            top_raw_score=last.top_raw_score,
            trades_placed=int(trades_placed),
            raw_scores=last.raw_scores,
        )
        self._cycles[-1] = updated
        if trades_placed > 0:
            self._cycles_since_last_trade = 0
            self._cycles_since_last_fill = 0

    # ------------------------------------------------------------------
    # Evidence digest
    # ------------------------------------------------------------------

    def build_evidence(
        self,
        *,
        cfg: AppConfig,
        recent_realized_pnl: Iterable[dict[str, Any]],
        open_position_pnl_total: float,
    ) -> AutotuneEvidence:
        """Materialise the per-cycle evidence payload for the LLM prompt."""
        last_snap = self._cycles[-1] if self._cycles else None
        all_scores: list[float] = []
        for s in self._cycles:
            all_scores.extend(s.raw_scores)
        rolling_dist = _distribution(all_scores)
        this_cycle_dist = _distribution(last_snap.raw_scores) if last_snap else {}
        thresholds = {
            "min_signal_strength": float(cfg.strategy.min_signal_strength),
            "min_exit_strength": float(cfg.strategy.min_exit_strength),
        }
        weights = {
            "weight_sma_cross": float(cfg.strategy.weight_sma_cross),
            "weight_ema_cross": float(cfg.strategy.weight_ema_cross),
            "weight_rsi": float(cfg.strategy.weight_rsi),
            "weight_macd": float(cfg.strategy.weight_macd),
            "weight_bollinger": float(cfg.strategy.weight_bollinger),
            "weight_donchian": float(cfg.strategy.weight_donchian),
            "weight_momentum": float(cfg.strategy.weight_momentum),
        }
        ev = AutotuneEvidence(
            cycle_index=int(last_snap.cycle_index) if last_snap else 0,
            tracked_count=int(last_snap.tracked) if last_snap else 0,
            candidates_this_cycle=int(last_snap.candidates) if last_snap else 0,
            top_raw_score=last_snap.top_raw_score if last_snap else None,
            raw_score_distribution={
                **{f"this_cycle.{k}": v for k, v in this_cycle_dist.items()},
                **{f"rolling.{k}": v for k, v in rolling_dist.items()},
            },
            cycles_since_last_candidate=int(self._cycles_since_last_candidate),
            cycles_since_last_trade=int(self._cycles_since_last_trade),
            cycles_since_last_fill=int(self._cycles_since_last_fill),
            last_n_cycles=[s.to_summary() for s in self._cycles],
            recent_realized_pnl=list(recent_realized_pnl),
            open_position_pnl_total=float(open_position_pnl_total),
            previous_tunings=list(self._tunings),
            current_thresholds=thresholds,
            current_weights=weights,
            current_spread_max_pct=float(cfg.tools.spread_max_pct),
        )
        return ev

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        request: TuneRequest,
        *,
        cfg: AppConfig,
    ) -> list[TuneApplied]:
        """Mutate ``cfg`` in place + persist + log. Returns the applied diff.

        No-op when ``request`` is empty. Changes that would assign the
        SAME value are dropped (no point logging a no-op edit).
        """
        if request.is_empty:
            return []
        applied: list[TuneApplied] = []
        for change in request.changes:
            section_obj = self._section_obj(cfg, change.section)
            if section_obj is None:
                continue
            if not hasattr(section_obj, change.field):
                continue
            previous = getattr(section_obj, change.field)
            if previous == change.value:
                continue
            try:
                setattr(section_obj, change.field, change.value)
            except (AttributeError, TypeError) as exc:
                self._log.warning(
                    "[autotune] failed to set %s.%s=%r: %s",
                    change.section, change.field, change.value, exc,
                )
                continue
            self._persist_field(change.section, change.field, change.value)
            applied.append(TuneApplied(
                section=change.section,
                field=change.field,
                previous=previous,
                current=change.value,
                rationale=change.rationale,
            ))

        if applied:
            self._tunings.append({
                "timestamp_unix": time.time(),
                "reason": request.reason,
                "changes": [a.to_dict() for a in applied],
            })
            self._log.info(
                "[autotune] applied %d change(s): %s — reason: %s",
                len(applied), render_tune_diff(applied), request.reason or "(no reason)",
            )
        return applied

    # ------------------------------------------------------------------
    # Persistence (rolling counters / tuning log) — for restart resume
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot for ``bot_state.json``."""
        return {
            "cycles_since_last_candidate": int(self._cycles_since_last_candidate),
            "cycles_since_last_trade": int(self._cycles_since_last_trade),
            "cycles_since_last_fill": int(self._cycles_since_last_fill),
            "tunings": list(self._tunings),
        }

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        try:
            self._cycles_since_last_candidate = int(payload.get("cycles_since_last_candidate") or 0)
            self._cycles_since_last_trade = int(payload.get("cycles_since_last_trade") or 0)
            self._cycles_since_last_fill = int(payload.get("cycles_since_last_fill") or 0)
            tunings = payload.get("tunings") or []
            if isinstance(tunings, list):
                self._tunings = deque(tunings, maxlen=self._tunings.maxlen)
        except (TypeError, ValueError) as exc:
            self._log.warning("[autotune] state restore failed: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _section_obj(cfg: AppConfig, section: str) -> Any:
        if section == "strategy":
            return cfg.strategy
        if section == "tools":
            return cfg.tools
        return None

    def _persist_field(self, section: str, field_name: str, value: Any) -> None:
        if self._store is None:
            return
        try:
            self._store.set_field(section, field_name, value)
        except Exception as exc:  # noqa: BLE001 - persistence never blocks tuning
            self._log.warning(
                "[autotune] persist failed (%s.%s=%r): %s",
                section, field_name, value, exc,
            )
