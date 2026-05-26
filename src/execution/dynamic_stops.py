"""Per-position dynamic stop-loss / take-profit overrides.

eToro's Public API does not expose a "modify position SL/TP" endpoint
— the only way to change a position's exit rules after open is to
close-and-reopen, which loses the entry timestamp + has tax/perf-
attribution side effects we want to avoid. Instead the bot enforces
SL/TP **client-side** via :meth:`PositionMonitor.positions_needing_close`,
which fires a synthetic CLOSE order when live mid breaches the
configured bands.

This module holds the per-position bands. When the manager LLM
emits a ``MODIFY_STOPS`` action it writes here; the monitor reads
from here on every cycle, falling back to the global guardrail
defaults when no per-position override is set.

Trailing support
----------------
When ``trailing_stop_pct`` is set on a position, the monitor ratchets
``stop_loss_pct`` upward as the position's MFE peak grows: the SL
trails the high-water mark by ``trailing_stop_pct`` percentage points
without ever moving wider. The base ``stop_loss_pct`` is the floor —
the trailing SL never moves looser than it.

Persistence
-----------
State serialises to a tiny dict; :class:`StatePersistence` writes it
to ``bot_state.json`` alongside the rest of session state so a restart
preserves whatever bands the LLM has set.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StopBand:
    """The per-position SL/TP override.

    All percentages are positive numbers expressed as percent (5.0 = 5%).
    ``stop_loss_pct`` is measured against the entry; ``take_profit_pct``
    against the entry. ``trailing_stop_pct`` (optional) makes the SL
    trail the position's MFE peak.
    """

    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float | None = None
    # Internal: highest favourable P/L pct we've ever seen on this
    # position. Used to anchor trailing-stop ratcheting. Stored in
    # PERCENT units, not USD.
    mfe_pct: float = 0.0
    # Optional free-text rationale the LLM emitted alongside the action
    # — surfaced in the MODIFY_STOPS alert and on /stats.
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "mfe_pct": self.mfe_pct,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StopBand":
        return cls(
            stop_loss_pct=float(d.get("stop_loss_pct") or 0.0),
            take_profit_pct=float(d.get("take_profit_pct") or 0.0),
            trailing_stop_pct=(
                float(d["trailing_stop_pct"])
                if d.get("trailing_stop_pct") is not None
                else None
            ),
            mfe_pct=float(d.get("mfe_pct") or 0.0),
            rationale=str(d.get("rationale") or ""),
        )


class DynamicStopsStore:
    """Thread-safe registry of per-position SL/TP overrides.

    The cycle thread and the controller thread both touch this; a
    simple lock keeps reads consistent with the trailing ratchet.
    """

    def __init__(
        self,
        *,
        default_stop_loss_pct: float,
        default_take_profit_pct: float,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._default_sl = float(default_stop_loss_pct)
        self._default_tp = float(default_take_profit_pct)
        self._lock = threading.Lock()
        self._bands: dict[int, StopBand] = {}
        self._log = logger or logging.getLogger("etrader.execution.dynamic_stops")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def effective_band(self, position_id: int) -> StopBand:
        """Return the SL/TP band the monitor should apply to a position.

        Falls back to the global guardrail defaults when no override is
        set. The returned StopBand is a *copy* — mutations don't affect
        the store.
        """
        with self._lock:
            band = self._bands.get(int(position_id))
            if band is None:
                return StopBand(
                    stop_loss_pct=self._default_sl,
                    take_profit_pct=self._default_tp,
                )
            return StopBand(
                stop_loss_pct=band.stop_loss_pct,
                take_profit_pct=band.take_profit_pct,
                trailing_stop_pct=band.trailing_stop_pct,
                mfe_pct=band.mfe_pct,
                rationale=band.rationale,
            )

    def has_override(self, position_id: int) -> bool:
        with self._lock:
            return int(position_id) in self._bands

    def snapshot(self) -> dict[int, dict[str, Any]]:
        """Whole-store view for telemetry / persistence."""
        with self._lock:
            return {pid: band.to_dict() for pid, band in self._bands.items()}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def set_band(
        self,
        position_id: int,
        *,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        rationale: str = "",
    ) -> StopBand:
        """Set / update a position's band.

        Any field passed as ``None`` keeps its previous value (or the
        global default if there was no previous band). Returns the
        resulting effective band.
        """
        pid = int(position_id)
        with self._lock:
            existing = self._bands.get(pid)
            sl = (
                float(stop_loss_pct) if stop_loss_pct is not None
                else (existing.stop_loss_pct if existing else self._default_sl)
            )
            tp = (
                float(take_profit_pct) if take_profit_pct is not None
                else (existing.take_profit_pct if existing else self._default_tp)
            )
            trail = (
                float(trailing_stop_pct) if trailing_stop_pct is not None
                else (existing.trailing_stop_pct if existing else None)
            )
            band = StopBand(
                stop_loss_pct=sl,
                take_profit_pct=tp,
                trailing_stop_pct=trail,
                mfe_pct=existing.mfe_pct if existing else 0.0,
                rationale=str(rationale or (existing.rationale if existing else "")),
            )
            self._bands[pid] = band
            return StopBand(**band.to_dict())  # return a copy

    def clear(self, position_id: int) -> None:
        with self._lock:
            self._bands.pop(int(position_id), None)

    def ratchet_trailing(
        self, position_id: int, *, current_pnl_pct: float,
    ) -> StopBand:
        """Update the band's trailing stop if MFE increases.

        Called by the monitor every cycle for each open position. When
        trailing is enabled and the position's current ``pnl_pct``
        exceeds the recorded MFE peak, the SL ratchets up to
        ``mfe_pct - trailing_stop_pct``. The base ``stop_loss_pct``
        acts as a floor — the trailing SL never moves looser than it.

        Returns the (possibly updated) effective band.
        """
        pid = int(position_id)
        with self._lock:
            band = self._bands.get(pid)
            if band is None or band.trailing_stop_pct is None:
                # No trailing configured: monitor uses static band.
                if band is None:
                    return StopBand(
                        stop_loss_pct=self._default_sl,
                        take_profit_pct=self._default_tp,
                    )
                return StopBand(**band.to_dict())
            # Update MFE peak in PERCENT units (trailing-SL math is
            # all done in pct relative to entry).
            if current_pnl_pct > band.mfe_pct:
                band.mfe_pct = float(current_pnl_pct)
                # Compute new dynamic SL: peak - trailing_stop_pct.
                # We invert the sign because SL is measured as the
                # *adverse* movement from entry. If MFE is +10% and
                # trailing is 4%, the effective SL becomes -(10-4)=-6%
                # below entry → i.e. the SL is now at +6% relative to
                # entry. We store this as a negative base SL (stop is
                # ABOVE entry once we're in profit). To keep
                # ``stop_loss_pct`` positive (the rest of the codebase
                # assumes positive %), we keep the convention by
                # storing the BREACH distance as a separate field:
                # see :attr:`effective_floor_pct`. For now, the
                # trailing-effective floor is folded into stop_loss_pct
                # by callers via :meth:`trailing_floor_pct`.
            return StopBand(**band.to_dict())

    def trailing_floor_pct(self, position_id: int) -> float | None:
        """Return the trailing SL's effective floor in PERCENT units.

        Signed (positive = above entry, negative = below entry):

        - If trailing isn't configured → ``None`` (monitor ignores the
          trailing path and falls back to the static SL).
        - Until MFE exceeds ``trailing_stop_pct`` → ``-stop_loss_pct``.
          Trailing only activates once there's enough profit to lock
          in (otherwise we'd tighten the SL with no profit cushion).
        - Once MFE > trailing_stop_pct, returns
          ``max(mfe_pct - trailing_stop_pct, -stop_loss_pct)``. The
          base floor is the absolute looseness ceiling — the floor
          can only TIGHTEN as MFE grows, never loosen below the
          base SL.
        """
        with self._lock:
            band = self._bands.get(int(position_id))
            if band is None or band.trailing_stop_pct is None:
                return None
            base_floor = -band.stop_loss_pct
            # Trailing only activates once MFE > trailing band.
            if band.mfe_pct <= band.trailing_stop_pct:
                return base_floor
            trailing = band.mfe_pct - band.trailing_stop_pct
            return max(trailing, base_floor)

    # ------------------------------------------------------------------
    # Persistence helpers — called by StatePersistence
    # ------------------------------------------------------------------

    def to_persistable(self) -> dict[str, Any]:
        return self.snapshot()

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        with self._lock:
            for k, v in payload.items():
                try:
                    pid = int(k)
                except (TypeError, ValueError):
                    continue
                if not isinstance(v, dict):
                    continue
                try:
                    self._bands[pid] = StopBand.from_dict(v)
                except (TypeError, ValueError):
                    continue
