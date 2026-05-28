"""Per-exchange session model.

The pre-refactor session model assumed every non-crypto, non-FX
instrument followed US equity hours (NY 09:30 → 16:00). That was wrong
for any user holding LSE, XETRA, Euronext, HKEX, TSE, ASX, TSX or
similar listings: the bot would gate them as "market closed" outside
NY hours and ``no_overnight`` would happily flatten them during their
own session.

This module owns the per-exchange schedule, keyed by the lowercased
``priceSource`` string eToro returns on the instrument metadata
(``nasdaq``, ``lse``, ``hkex``, …). Public API:

* :func:`session_for(meta, asset_class, now)` — the canonical lookup.
  Resolves the exchange via meta when possible, falls back to US
  equity hours when there's no usable meta. CRYPTO is always open,
  FX is 24x5.
* :func:`session_window_for(meta, asset_class, now)` — returns the
  ``(open_utc, close_utc)`` window for the calendar date of ``now``
  in the exchange's local timezone (used by the ``no_overnight``
  flatten-before-close logic).
* :func:`exchange_label(meta)` — human-readable exchange string for
  log lines and Telegram alerts.

Limitations (called out so we don't pretend they're handled):

* Holiday calendars per exchange are NOT modelled. A bank holiday on
  LSE will look like a normal trading day to this module.
* Intraday lunch breaks (TSE, HKEX, SSE/SZSE) are NOT modelled —
  treated as one continuous session for the day. eToro routes such
  trades anyway; this is a coarse availability check, not an
  execution-quality model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..strategy.tools.base import AssetClass


# ---------------------------------------------------------------------------
# Exchange registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExchangeSchedule:
    """Local-time open/close + IANA timezone for one exchange."""

    label: str            # short human-readable name ("NYSE", "LSE", …)
    tz: ZoneInfo
    open_local: time
    close_local: time


# Keyed by lowercased ``priceSource`` (eToro's instrument metadata
# carries one of these strings; see :data:`src.strategy.tools.base
# ._EQUITY_PRICE_SOURCES`). Lunch breaks rolled into a single window
# because the bot only uses this for availability gating, not
# execution-quality modelling.
_EXCHANGE_HOURS: dict[str, ExchangeSchedule] = {
    # North America — NY local
    "nasdaq":   ExchangeSchedule("NASDAQ", ZoneInfo("America/New_York"),  time(9, 30), time(16, 0)),
    "nyse":     ExchangeSchedule("NYSE",   ZoneInfo("America/New_York"),  time(9, 30), time(16, 0)),
    "amex":     ExchangeSchedule("AMEX",   ZoneInfo("America/New_York"),  time(9, 30), time(16, 0)),
    "arca":     ExchangeSchedule("ARCA",   ZoneInfo("America/New_York"),  time(9, 30), time(16, 0)),
    "bats":     ExchangeSchedule("BATS",   ZoneInfo("America/New_York"),  time(9, 30), time(16, 0)),
    "tsx":      ExchangeSchedule("TSX",    ZoneInfo("America/Toronto"),   time(9, 30), time(16, 0)),
    "tsxv":     ExchangeSchedule("TSXV",   ZoneInfo("America/Toronto"),   time(9, 30), time(16, 0)),
    # UK + Ireland
    "lse":      ExchangeSchedule("LSE",    ZoneInfo("Europe/London"),     time(8, 0),  time(16, 30)),
    # Continental Europe
    "fwb":      ExchangeSchedule("FWB",    ZoneInfo("Europe/Berlin"),     time(9, 0),  time(17, 30)),
    "xetra":    ExchangeSchedule("XETRA",  ZoneInfo("Europe/Berlin"),     time(9, 0),  time(17, 30)),
    "euronext": ExchangeSchedule("Euronext", ZoneInfo("Europe/Paris"),    time(9, 0),  time(17, 30)),
    "milan":    ExchangeSchedule("Milan",  ZoneInfo("Europe/Rome"),       time(9, 0),  time(17, 30)),
    "madrid":   ExchangeSchedule("Madrid", ZoneInfo("Europe/Madrid"),     time(9, 0),  time(17, 30)),
    "swx":      ExchangeSchedule("SIX",    ZoneInfo("Europe/Zurich"),     time(9, 0),  time(17, 30)),
    "borsa":    ExchangeSchedule("BIST",   ZoneInfo("Europe/Istanbul"),   time(10, 0), time(18, 0)),
    # Asia / Pacific
    "tse":      ExchangeSchedule("TSE",    ZoneInfo("Asia/Tokyo"),        time(9, 0),  time(15, 30)),
    "hkex":     ExchangeSchedule("HKEX",   ZoneInfo("Asia/Hong_Kong"),    time(9, 30), time(16, 0)),
    "sse":      ExchangeSchedule("SSE",    ZoneInfo("Asia/Shanghai"),     time(9, 30), time(15, 0)),
    "szse":     ExchangeSchedule("SZSE",   ZoneInfo("Asia/Shanghai"),     time(9, 30), time(15, 0)),
    "asx":      ExchangeSchedule("ASX",    ZoneInfo("Australia/Sydney"),  time(10, 0), time(16, 0)),
}


# Default fallback when we have no usable meta — matches the old
# behaviour (NY 09:30 → 16:00) so callers without meta aren't
# silently downgraded to "always closed".
_DEFAULT_EQUITY = _EXCHANGE_HOURS["nyse"]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionState:
    """Snapshot of "is this asset's market open right now?"."""

    is_open: bool
    seconds_since_open: int | None
    next_open_utc: datetime | None
    exchange_label: str           # "NYSE" / "LSE" / "CRYPTO" / "FX" / …


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

def session_for(
    meta: Any | None,
    asset_class: AssetClass,
    now: datetime,
) -> SessionState:
    """Snapshot for the given instrument + asset class.

    Resolution order:

    1. CRYPTO → always open.
    2. FX → 24x5 (Mon-Fri UTC).
    3. ``meta.price_source`` matches a known exchange → that
       exchange's schedule.
    4. Anything else → US equity hours (NY 09:30 → 16:00).
    """
    now_utc = _to_utc(now)
    if asset_class == AssetClass.CRYPTO:
        return SessionState(True, None, None, exchange_label="CRYPTO")
    if asset_class == AssetClass.FX:
        return _fx_state(now_utc)
    schedule = _schedule_for_meta(meta) or _DEFAULT_EQUITY
    return _exchange_state(now_utc, schedule)


def session_window_for(
    meta: Any | None,
    asset_class: AssetClass,
    now: datetime,
) -> tuple[datetime, datetime] | None:
    """Return ``(open_utc, close_utc)`` for the *calendar day* of ``now``
    in the instrument's local timezone.

    Returns ``None`` for assets that don't have a fixed daily window
    (CRYPTO is 24/7, FX is treated as a continuous weekday session).
    """
    if asset_class in (AssetClass.CRYPTO, AssetClass.FX):
        return None
    schedule = _schedule_for_meta(meta) or _DEFAULT_EQUITY
    return _window(_to_utc(now), schedule)


def exchange_label(meta: Any | None, asset_class: AssetClass) -> str:
    """Short label for log lines / Telegram alerts.

    ``CRYPTO``, ``FX``, or the exchange code (``NASDAQ`` / ``LSE`` / …).
    """
    if asset_class == AssetClass.CRYPTO:
        return "CRYPTO"
    if asset_class == AssetClass.FX:
        return "FX"
    schedule = _schedule_for_meta(meta) or _DEFAULT_EQUITY
    return schedule.label


def resolve_exchange_label_for(
    meta: Any | None,
    symbol: str,
) -> str | None:
    """Return the exchange label for a (meta, symbol) pair, or ``None``
    if the meta is missing entirely.

    Thin wrapper that classifies the instrument via the same asset-class
    helper used by the rest of the strategy code, so log lines, LLM
    payloads, and ``no_overnight`` flatten decisions all label the
    instrument identically.
    """
    if meta is None:
        return None
    # Lazy import — ``asset_class_for`` lives under ``strategy.tools``,
    # which itself imports from ``execution``. Doing the import at
    # module load creates a cycle.
    from ..strategy.tools.base import asset_class_for

    cls = asset_class_for(meta, symbol=symbol)
    return exchange_label(meta, cls)


def currently_open_exchange_labels(now: datetime | None = None) -> list[str]:
    """Return the de-duplicated list of exchange labels that are *open*
    in the equity sense at ``now`` (defaults to current UTC).

    Used by the universe-rotation LLM call so it can bias nominations
    toward markets the bot can actually trade right now. CRYPTO is
    always included (24/7); FX is included on weekdays.

    Implementation is intentionally a linear scan over the static
    registry — there are ~20 entries and this is called at most once
    per cycle.
    """
    now_utc = _to_utc(now if now is not None else datetime.now(timezone.utc))
    seen: set[str] = set()
    open_labels: list[str] = []
    for sched in _EXCHANGE_HOURS.values():
        if sched.label in seen:
            continue
        state = _exchange_state(now_utc, sched)
        if state.is_open:
            seen.add(sched.label)
            open_labels.append(sched.label)
    open_labels.append("CRYPTO")
    if now_utc.weekday() < 5:
        open_labels.append("FX")
    return open_labels


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _schedule_for_meta(meta: Any | None) -> ExchangeSchedule | None:
    if meta is None:
        return None
    src = getattr(meta, "price_source", None)
    if not src:
        return None
    return _EXCHANGE_HOURS.get(str(src).strip().lower())


def _exchange_state(now_utc: datetime, sched: ExchangeSchedule) -> SessionState:
    open_utc, close_utc = _window(now_utc, sched)
    now_local = now_utc.astimezone(sched.tz)
    is_weekday = now_local.weekday() < 5
    if is_weekday and open_utc <= now_utc < close_utc:
        elapsed = int((now_utc - open_utc).total_seconds())
        return SessionState(
            is_open=True,
            seconds_since_open=elapsed,
            next_open_utc=open_utc,
            exchange_label=sched.label,
        )
    return SessionState(
        is_open=False,
        seconds_since_open=None,
        next_open_utc=_next_open(now_utc, sched),
        exchange_label=sched.label,
    )


def _window(now_utc: datetime, sched: ExchangeSchedule) -> tuple[datetime, datetime]:
    """Open/close UTC for the local-time calendar day of ``now_utc``."""
    now_local = now_utc.astimezone(sched.tz)
    open_local = now_local.replace(
        hour=sched.open_local.hour,
        minute=sched.open_local.minute,
        second=0,
        microsecond=0,
    )
    close_local = now_local.replace(
        hour=sched.close_local.hour,
        minute=sched.close_local.minute,
        second=0,
        microsecond=0,
    )
    return (
        open_local.astimezone(timezone.utc),
        close_local.astimezone(timezone.utc),
    )


def _next_open(now_utc: datetime, sched: ExchangeSchedule) -> datetime:
    """Next weekday session open (in UTC), strictly *after* ``now_utc``."""
    now_local = now_utc.astimezone(sched.tz)
    candidate = now_local.replace(
        hour=sched.open_local.hour,
        minute=sched.open_local.minute,
        second=0,
        microsecond=0,
    )
    if candidate.astimezone(timezone.utc) <= now_utc:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def _fx_state(now_utc: datetime) -> SessionState:
    weekday = now_utc.weekday()
    if weekday < 5:
        return SessionState(
            is_open=True,
            seconds_since_open=_seconds_since_midnight(now_utc),
            next_open_utc=now_utc.replace(hour=0, minute=0, second=0, microsecond=0),
            exchange_label="FX",
        )
    days_to_monday = 7 - weekday
    next_open = (now_utc + timedelta(days=days_to_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return SessionState(
        is_open=False,
        seconds_since_open=None,
        next_open_utc=next_open,
        exchange_label="FX",
    )


def _to_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _seconds_since_midnight(now_utc: datetime) -> int:
    midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((now_utc - midnight).total_seconds())


__all__ = [
    "ExchangeSchedule",
    "SessionState",
    "currently_open_exchange_labels",
    "exchange_label",
    "resolve_exchange_label_for",
    "session_for",
    "session_window_for",
]
