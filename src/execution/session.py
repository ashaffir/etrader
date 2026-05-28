"""Session-awareness helpers for the order monitor.

This module is the thin back-compat layer over
:mod:`src.execution.exchange_session`. The richer per-exchange API
lives there; everything below preserves the legacy signature
``session_state(asset_class, now)`` so older call sites (stuck-order
monitor, tests) keep working unchanged.

Two questions the monitor needs to answer for any pending order:

1. **Is the asset's market open right now?**  If not, the order is
   *supposed* to sit pending; don't even consider cancelling it.
2. **If the market is open, how long has it been open?**  We give an
   order ``grace_seconds_after_open`` to fill before declaring it
   stuck.

When the caller has the instrument metadata, prefer the meta-aware
:func:`src.execution.exchange_session.session_for` directly — that
function picks the right exchange (LSE, HKEX, XETRA, …) for the
instrument. The legacy ``session_state(asset_class, now)`` here
keeps the previous behaviour: US equity hours for any non-crypto,
non-FX asset, which is the safe default when no meta is available.

US equity hours are DST-aware (anchored on NY 09:30 → 16:00 ET, so
the UTC window shifts 13:30→20:00 (EDT) ↔ 14:30→21:00 (EST) on its
own).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from . import exchange_session as _ex
from ..strategy.tools.base import AssetClass


# Re-exported so callers don't import from two modules for one type.
SessionState = _ex.SessionState


def session_state(asset_class: AssetClass, now: datetime) -> SessionState:
    """Legacy meta-less lookup. Falls back to US equity hours for
    every non-crypto, non-FX asset.

    Prefer :func:`session_for_meta` when the caller has the
    instrument metadata — the bot supports LSE, XETRA, HKEX, TSE,
    ASX, … instruments and meta-less lookup ignores those exchanges.
    """
    return _ex.session_for(None, asset_class, now)


def session_for_meta(
    meta: Any | None, asset_class: AssetClass, now: datetime,
) -> SessionState:
    """Meta-aware lookup — re-export of :func:`exchange_session.session_for`."""
    return _ex.session_for(meta, asset_class, now)


def is_market_open(asset_class: AssetClass, now: datetime) -> bool:
    """Convenience boolean — see :func:`session_state` for full info."""
    return session_state(asset_class, now).is_open


def seconds_since_open(asset_class: AssetClass, now: datetime) -> int | None:
    """Seconds elapsed since the current session's open, or ``None`` if closed.

    Returns ``None`` for closed sessions AND for "always open" assets
    (crypto) — callers should treat ``None`` as "session model doesn't
    apply, use absolute placement age instead".
    """
    return session_state(asset_class, now).seconds_since_open


def equity_session_window(now: datetime) -> tuple[datetime, datetime]:
    """Return ``(open_utc, close_utc)`` for the US equity session on ``now``'s date.

    Kept for back-compat. New code should use
    :func:`exchange_session.session_window_for` which picks the right
    exchange from the instrument metadata.
    """
    window = _ex.session_window_for(None, AssetClass.STOCK, now)
    assert window is not None  # STOCK always has a daily window
    return window


__all__ = [
    "SessionState",
    "session_state",
    "session_for_meta",
    "is_market_open",
    "seconds_since_open",
    "equity_session_window",
]
