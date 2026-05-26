"""Threshold-triggered review of open bot-owned positions.

The decision LLM runs every cycle but doesn't necessarily re-consider
every existing position. To keep cost low while still being
responsive when a position misbehaves, we pre-screen each open
position against a small set of triggers. Positions that fire a
trigger are flagged in the decision prompt with the reason, so the
LLM is forced to attend to them.

Triggers (configurable in ``[position_review]`` of ``config.toml``):

- **drawdown**:  ``pnl_pct <= -drawdown_pct``. Trade is bleeding past
  a configurable threshold (default -2%).
- **trailing_pullback**:  the position has given back at least
  ``pullback_pct`` percent of its peak P/L (in PCT space). E.g. peak
  was +6%, current is +2% → pullback of 4pp. Catches "winner reversing".
- **stale_hold**:  ``time_held > stale_hold_minutes`` AND
  ``|pnl_pct| < stale_threshold_pct``. Trade has gone nowhere for too
  long; opportunity cost trigger.
- **time_to_close**:  ``time_held > max_hold_minutes`` regardless of
  P/L — hard ceiling on how long the bot may hold a single trade.

The review is *advisory* — it never closes anything by itself. It
only annotates the LLM context. The LLM owns the final decision.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..etoro.trading import Position


@dataclass(frozen=True)
class PositionReviewConfig:
    """Per-trigger thresholds. All percentages are positive numbers.

    ``0`` or ``None`` disables a trigger.
    """

    drawdown_pct: float = 2.0
    pullback_pct: float = 3.0
    stale_hold_minutes: float = 60.0
    stale_threshold_pct: float = 0.5
    max_hold_minutes: float = 240.0  # 4h default ceiling

    @classmethod
    def from_mapping(cls, m: Mapping[str, object] | None) -> "PositionReviewConfig":
        if not m:
            return cls()
        kw: dict[str, float] = {}
        for k, v in m.items():
            if v is None:
                continue
            try:
                kw[str(k)] = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return cls(**{k: kw[k] for k in kw if k in cls.__dataclass_fields__})


@dataclass
class PositionReview:
    """One review annotation for the LLM."""

    position_id: int
    instrument_id: int
    symbol: str
    pnl_usd: float
    pnl_pct: float
    mfe_usd: float
    mae_usd: float
    time_held_seconds: int
    triggers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "position_id": self.position_id,
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "pnl_usd": round(self.pnl_usd, 4),
            "pnl_pct": round(self.pnl_pct, 4),
            "mfe_usd": round(self.mfe_usd, 4),
            "mae_usd": round(self.mae_usd, 4),
            "time_held_seconds": self.time_held_seconds,
            "time_held_minutes": round(self.time_held_seconds / 60.0, 1),
            "triggers": list(self.triggers),
            "notes": list(self.notes),
        }


class PositionReviewer:
    """Stateless evaluator — checks a snapshot of positions against the config."""

    def __init__(
        self,
        cfg: PositionReviewConfig,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._cfg = cfg
        self._log = logger or logging.getLogger("etrader.strategy.position_review")

    @property
    def config(self) -> PositionReviewConfig:
        return self._cfg

    def evaluate(
        self,
        *,
        bot_owned_positions: Iterable[Position],
        symbol_for_id: Mapping[int, str],
        perf_open_states: Mapping[int, "object"] | None = None,
        live_rates: Mapping[int, "object"] | None = None,
        now_epoch: float | None = None,
    ) -> list[PositionReview]:
        """Run every trigger against every open bot-owned position.

        ``perf_open_states`` is the dict the PerformanceTracker exposes
        via :py:attr:`open_states` — keyed by position_id, value has
        ``last_pnl_usd``, ``last_pnl_pct``, ``mfe_usd``, ``mae_usd``,
        and ``opened_at_iso``.

        ``live_rates`` is the same dict :class:`PositionMonitor` consumes;
        used as a fallback when the tracker hasn't observed a position
        yet (first cycle after open).
        """
        perf = dict(perf_open_states or {})
        now = float(now_epoch if now_epoch is not None else time.time())
        out: list[PositionReview] = []
        for pos in bot_owned_positions:
            review = self._evaluate_one(
                pos=pos,
                symbol_for_id=symbol_for_id,
                perf=perf,
                live_rates=live_rates or {},
                now=now,
            )
            if review.triggers:
                out.append(review)
        return out

    def _evaluate_one(
        self,
        *,
        pos: Position,
        symbol_for_id: Mapping[int, str],
        perf: Mapping[int, "object"],
        live_rates: Mapping[int, "object"],
        now: float,
    ) -> PositionReview:
        symbol = symbol_for_id.get(pos.instrument_id, f"INST-{pos.instrument_id}")
        snap = perf.get(pos.position_id)
        pnl_usd, pnl_pct, mfe_usd, mae_usd, opened_epoch = self._extract_perf(
            pos=pos, snap=snap, live_rates=live_rates, now=now,
        )
        time_held = int(max(0.0, now - opened_epoch))
        review = PositionReview(
            position_id=pos.position_id,
            instrument_id=pos.instrument_id,
            symbol=symbol,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            mfe_usd=mfe_usd,
            mae_usd=mae_usd,
            time_held_seconds=time_held,
        )
        self._apply_triggers(review)
        return review

    def _extract_perf(
        self,
        *,
        pos: Position,
        snap: object | None,
        live_rates: Mapping[int, object],
        now: float,
    ) -> tuple[float, float, float, float, float]:
        """Return ``(pnl_usd, pnl_pct, mfe_usd, mae_usd, opened_epoch)``.

        Prefers the tracker's running mark when available, falls back to
        ``pos.pnl`` plus a synthetic pct computed from ``live_rates``.
        """
        opened_epoch = now  # if we don't have an open time, treat as fresh.
        if snap is not None:
            pnl_usd = float(getattr(snap, "last_pnl_usd", None) or pos.pnl or 0.0)
            pnl_pct = float(getattr(snap, "last_pnl_pct", None) or 0.0)
            mfe_usd = float(getattr(snap, "mfe_usd", None) or 0.0)
            mae_usd = float(getattr(snap, "mae_usd", None) or 0.0)
            opened_iso = str(getattr(snap, "opened_at_iso", "") or "")
            opened_epoch = self._iso_to_epoch(opened_iso, fallback=now)
            return pnl_usd, pnl_pct, mfe_usd, mae_usd, opened_epoch
        # Fallback when tracker hasn't seen this position yet: derive
        # pct from current live rate so the first cycle still has data.
        pnl_usd = float(pos.pnl or 0.0)
        pnl_pct = self._derive_pct(pos, live_rates)
        return pnl_usd, pnl_pct, 0.0, 0.0, opened_epoch

    @staticmethod
    def _iso_to_epoch(iso: str, *, fallback: float) -> float:
        if not iso:
            return fallback
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.timestamp()
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _derive_pct(pos: Position, live_rates: Mapping[int, object]) -> float:
        if pos.open_rate <= 0:
            return 0.0
        rate = live_rates.get(pos.instrument_id)
        mid = getattr(rate, "mid", None)
        if mid is None:
            return 0.0
        try:
            mid_f = float(mid)
        except (TypeError, ValueError):
            return 0.0
        pct = (mid_f - pos.open_rate) / pos.open_rate * 100.0
        if not pos.is_buy:
            pct = -pct
        return pct

    def _apply_triggers(self, r: PositionReview) -> None:
        cfg = self._cfg
        if cfg.drawdown_pct > 0 and r.pnl_pct <= -cfg.drawdown_pct:
            r.triggers.append("drawdown")
            r.notes.append(
                f"P/L {r.pnl_pct:+.2f}% breached drawdown cap "
                f"-{cfg.drawdown_pct:.2f}%"
            )
        # Trailing-pullback: only meaningful when MFE was positive.
        if cfg.pullback_pct > 0 and r.mfe_usd > 0 and r.pnl_usd > 0:
            # Express both in pct-of-MFE terms when we have the inputs.
            pullback_usd = r.mfe_usd - r.pnl_usd
            pullback_pct_of_mfe = (
                pullback_usd / r.mfe_usd * 100.0 if r.mfe_usd > 0 else 0.0
            )
            if pullback_pct_of_mfe >= cfg.pullback_pct:
                r.triggers.append("trailing_pullback")
                r.notes.append(
                    f"Gave back ${pullback_usd:+.2f} ({pullback_pct_of_mfe:.1f}% "
                    f"of MFE ${r.mfe_usd:+.2f}); pullback cap {cfg.pullback_pct:.1f}%"
                )
        if cfg.stale_hold_minutes > 0 and cfg.stale_threshold_pct >= 0:
            stale_seconds = int(cfg.stale_hold_minutes * 60)
            if (
                r.time_held_seconds >= stale_seconds
                and abs(r.pnl_pct) < cfg.stale_threshold_pct
            ):
                r.triggers.append("stale_hold")
                r.notes.append(
                    f"Flat ({r.pnl_pct:+.2f}%) for "
                    f"{r.time_held_seconds // 60} min ≥ "
                    f"{cfg.stale_hold_minutes:.0f} min"
                )
        if cfg.max_hold_minutes > 0:
            ceiling = int(cfg.max_hold_minutes * 60)
            if r.time_held_seconds >= ceiling:
                r.triggers.append("max_hold")
                r.notes.append(
                    f"Held for {r.time_held_seconds // 60} min ≥ "
                    f"{cfg.max_hold_minutes:.0f} min ceiling"
                )
