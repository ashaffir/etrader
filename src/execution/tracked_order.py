"""Small dataclass shared between :mod:`monitor` and :mod:`stuck_orders`.

Carved into its own module to break what would otherwise be a one-way
import cycle (``monitor`` -> ``stuck_orders`` -> ``monitor``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..strategy.tools.base import AssetClass


@dataclass
class TrackedOrder:
    """An order this session placed and is waiting to see settle in /pnl.

    ``action`` is "BUY" (a market-open we expect to materialise as a
    new position) or "CLOSE" (a market-close we expect to remove an
    existing position from the portfolio).

    ``placed_at_utc`` is the wall-clock placement time, needed for
    session-aware grace calculations. ``placed_at_monotonic`` survives
    NTP slew for absolute grace fallbacks (crypto / FX off-hours).
    ``position_id`` is the position the CLOSE order targets — unused
    (==0) for BUY orders.
    """

    order_id: int
    instrument_id: int
    symbol: str
    action: str           # "BUY" | "CLOSE"
    asset_class: AssetClass
    amount_usd: float
    placed_at_utc: datetime
    placed_at_monotonic: float
    position_id: int = 0
