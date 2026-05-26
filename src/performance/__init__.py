"""Bot-attributable performance tracker.

Public API:

- :class:`PerformanceTracker` — the long-lived object the controller owns.
- :class:`OpenTradeState`, :class:`RealizedTrade`, :class:`DailySnapshot` — the records.

The cycle calls :meth:`PerformanceTracker.record_open` on a successful
BUY, :meth:`observe_positions` once per cycle with the live bot-owned
positions, and :meth:`record_close` from the monitor's reconcile when
a tracked position settles.
"""

from .tracker import PerformanceTracker
from .types import DailySnapshot, OpenTradeState, RealizedTrade

__all__ = [
    "PerformanceTracker",
    "OpenTradeState",
    "RealizedTrade",
    "DailySnapshot",
]
