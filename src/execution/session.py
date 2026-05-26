"""Session-awareness helpers for the order monitor.

Pure functions — no I/O, no clock dependency beyond the ``now`` arg —
so they're easy to unit-test deterministically.

Two questions the monitor needs to answer for any pending order:

1. **Is the asset's market open right now?**  If not, the order is
   *supposed* to sit pending; don't even consider cancelling it.
2. **If the market is open, how long has it been open?**  We give an
   order ``grace_seconds_after_open`` to fill before declaring it
   stuck. This protects the bot against false-positive cancellations
   in the first few seconds of the session when the broker may still
   be matching the order.

The heuristics here intentionally match
:class:`src.strategy.tools.context_tools.MarketHoursTool` so the same
view of "market hours" is used everywhere:

* **Crypto**:  24x7 — always open.
* **FX**:      24x5 — closed weekends (Sat/Sun UTC).
* **Stocks / ETFs / indices / commodities / other**:  weekdays UTC
  13:30–21:00. This is the US-equity regular-session window in
  Eastern Daylight Time; the bot is currently US-equity-centric. DST
  edges and trading holidays are not modelled — a separate enhancement
  if/when we need that fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from ..strategy.tools.base import AssetClass


# US equity regular session in UTC (EDT). Crude but matches MarketHoursTool.
_EQUITY_OPEN_UTC = time(hour=13, minute=30)
_EQUITY_CLOSE_UTC = time(hour=21, minute=0)


@dataclass(frozen=True)
class SessionState:
    """Snapshot of "is this asset's market open right now?".

    ``seconds_since_open`` is ``None`` when the market is closed.
    ``next_open_utc`` is ``None`` for assets that are always open
    (crypto), and otherwise the UTC datetime of the next session open
    (in the future if currently closed, or *today's* open if currently
    open — useful for measuring elapsed in-session time uniformly).
    """

    is_open: bool
    seconds_since_open: int | None
    next_open_utc: datetime | None


def session_state(asset_class: AssetClass, now: datetime) -> SessionState:
    """Snapshot for the given asset class.

    ``now`` must be timezone-aware (UTC recommended). We coerce it to
    UTC up front so callers don't have to.
    """
    now_utc = _to_utc(now)
    if asset_class == AssetClass.CRYPTO:
        return SessionState(is_open=True, seconds_since_open=None, next_open_utc=None)

    if asset_class == AssetClass.FX:
        return _fx_state(now_utc)

    return _equity_state(now_utc)


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


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _to_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _fx_state(now_utc: datetime) -> SessionState:
    weekday = now_utc.weekday()  # Mon=0..Sun=6
    if weekday < 5:
        return SessionState(
            is_open=True,
            seconds_since_open=_seconds_since_midnight(now_utc),
            next_open_utc=now_utc.replace(hour=0, minute=0, second=0, microsecond=0),
        )
    # Weekend → next open is Monday 00:00 UTC.
    days_to_monday = 7 - weekday
    next_open = (now_utc + timedelta(days=days_to_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return SessionState(is_open=False, seconds_since_open=None, next_open_utc=next_open)


def _equity_state(now_utc: datetime) -> SessionState:
    weekday = now_utc.weekday()
    open_today = now_utc.replace(
        hour=_EQUITY_OPEN_UTC.hour,
        minute=_EQUITY_OPEN_UTC.minute,
        second=0,
        microsecond=0,
    )
    close_today = now_utc.replace(
        hour=_EQUITY_CLOSE_UTC.hour,
        minute=_EQUITY_CLOSE_UTC.minute,
        second=0,
        microsecond=0,
    )
    if weekday < 5 and open_today <= now_utc < close_today:
        elapsed = int((now_utc - open_today).total_seconds())
        return SessionState(is_open=True, seconds_since_open=elapsed, next_open_utc=open_today)

    return SessionState(
        is_open=False,
        seconds_since_open=None,
        next_open_utc=_next_equity_open(now_utc),
    )


def _next_equity_open(now_utc: datetime) -> datetime:
    """Next weekday UTC 13:30 strictly *after* ``now_utc``."""
    candidate = now_utc.replace(
        hour=_EQUITY_OPEN_UTC.hour,
        minute=_EQUITY_OPEN_UTC.minute,
        second=0,
        microsecond=0,
    )
    if candidate <= now_utc:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _seconds_since_midnight(now_utc: datetime) -> int:
    midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((now_utc - midnight).total_seconds())
