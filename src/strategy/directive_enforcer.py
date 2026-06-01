"""Code-level enforcement of operator directives.

Risk-layer hooks (:class:`Directives.is_symbol_blocked`,
``max_total_account_invested_usd``) live alongside the data class;
this module owns the slightly heavier logic the cycle needs:

* :func:`prescreen_candidates` — drop blocked symbols / sectors from
  the candidate list before the LLM sees them.
* :func:`build_directive_close_requests` — emit :class:`TradeRequest`
  CLOSE actions for positions that violate ``no_overnight`` or
  ``hold_ceiling_minutes``.

Both functions are pure (no I/O, no clock dependency beyond an
explicit ``now`` arg) so unit tests can drive them deterministically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable

from ..execution.exchange_session import (
    exchange_label,
    session_for,
    session_window_for,
)
from ..performance.types import OpenTradeState
from .directives import Directives
from .risk import TradeRequest
from .tools.base import AssetClass, asset_class_for


# Pre-bell cushion. When ``no_overnight`` is on AND the session is
# currently open, we start emitting CLOSE requests this many seconds
# before the bell rings so the orders land while there's still
# liquidity on the book. Once the session is *closed* the close-out
# fires every cycle regardless of this constant — see
# :func:`_should_flatten_equities` for the full rule.
DEFAULT_FLATTEN_WINDOW_SECONDS = 5 * 60


def prescreen_candidates(
    *,
    directives: Directives,
    candidates: Iterable[Any],
    fundamentals_lookup: Callable[[str], Any] | None = None,
) -> tuple[list[Any], list[tuple[str, str]]]:
    """Filter ``candidates`` against ``blocked_symbols`` + ``blocked_sectors``.

    Returns ``(kept, dropped)``. ``dropped`` is a list of
    ``(symbol, reason)`` tuples the cycle can log and surface to the
    LLM (so the operator knows *why* a name disappeared).

    ``fundamentals_lookup`` is an optional ``symbol → snapshot``
    callable; when supplied and ``blocked_sectors`` is non-empty,
    candidates whose ``snapshot.sector`` is blocked are dropped too.
    """
    kept: list[Any] = []
    dropped: list[tuple[str, str]] = []
    for cand in candidates:
        symbol = (getattr(cand, "symbol", "") or "").strip().upper()
        if not symbol:
            kept.append(cand)
            continue
        if directives.is_symbol_blocked(symbol):
            dropped.append((symbol, f"blocked_symbols ({symbol})"))
            continue
        if directives.blocked_sectors and fundamentals_lookup is not None:
            sector = _extract_sector(fundamentals_lookup, symbol)
            if sector and directives.is_sector_blocked(sector):
                dropped.append((symbol, f"blocked_sectors ({sector})"))
                continue
        kept.append(cand)
    return kept, dropped


def _extract_sector(
    fundamentals_lookup: Callable[[str], Any],
    symbol: str,
) -> str | None:
    try:
        snap = fundamentals_lookup(symbol)
    except Exception:  # noqa: BLE001 — fundamentals must never break enforcement
        return None
    if snap is None:
        return None
    sector = getattr(snap, "sector", None)
    if sector is None and isinstance(snap, dict):
        sector = snap.get("sector")
    return sector if isinstance(sector, str) and sector.strip() else None


# ---------------------------------------------------------------------------
# CLOSE injection for time/hold directives
# ---------------------------------------------------------------------------

def build_directive_close_requests(
    *,
    directives: Directives,
    bot_owned_positions: Iterable[Any],
    symbol_for_id: dict[int, str],
    instrument_metas: dict[int, Any],
    open_states: dict[int, OpenTradeState] | None = None,
    now: datetime,
    flatten_window_seconds: int = DEFAULT_FLATTEN_WINDOW_SECONDS,
    earnings_lookup: Any | None = None,
) -> tuple[list[TradeRequest], list[dict[str, Any]]]:
    """Build CLOSE requests for directive-driven exits.

    Returns ``(requests, notes)``:

    * ``requests``  — a list of :class:`TradeRequest` ready to be
      handed to the risk evaluator / executor (``close_fraction``
      omitted = full close).
    * ``notes`` — one structured dict per closed position describing
      which directive fired (for logging + the LLM payload).

    Rules:

    * ``hold_ceiling_minutes`` (when > 0): close any bot position
      whose elapsed hold time exceeds the ceiling.
    * ``no_overnight``: close any non-crypto bot position whenever
      the US equity session is currently closed (weekend, holiday,
      pre-market, after-hours) OR is open but within
      ``flatten_window_seconds`` of close. See
      :func:`_should_flatten_equities` for details.

    A position can match both rules — we deduplicate so the cycle
    only emits one CLOSE per position.
    """
    if open_states is None:
        open_states = {}

    requests: list[TradeRequest] = []
    notes: list[dict[str, Any]] = []
    seen: set[int] = set()

    hold_ceiling_seconds = (
        int(directives.hold_ceiling_minutes) * 60
        if directives.hold_ceiling_minutes > 0 else 0
    )

    for pos in bot_owned_positions:
        try:
            pid = int(getattr(pos, "position_id", 0) or 0)
            inst_id = int(getattr(pos, "instrument_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0 or pid in seen:
            continue
        symbol = symbol_for_id.get(inst_id, f"INST-{inst_id}")
        meta = instrument_metas.get(inst_id)
        asset_class = asset_class_for(meta, symbol=symbol)

        reason: str | None = None
        directive: str | None = None
        held_seconds = _held_seconds(pid, open_states, now=now)

        if hold_ceiling_seconds and held_seconds is not None and (
            held_seconds >= hold_ceiling_seconds
        ):
            directive = "hold_ceiling_minutes"
            reason = (
                f"held {held_seconds // 60} min ≥ "
                f"{directives.hold_ceiling_minutes} min ceiling"
            )
        elif directives.no_overnight and _position_needs_flatten(
            meta=meta,
            asset_class=asset_class,
            now=now,
            window_seconds=flatten_window_seconds,
        ):
            directive = "no_overnight"
            reason = _flatten_reason(
                meta=meta, asset_class=asset_class, now=now,
                window_seconds=flatten_window_seconds,
            )
        elif (
            directives.pre_earnings_close_hours > 0
            and earnings_lookup is not None
            and asset_class != AssetClass.CRYPTO
        ):
            entry = _safe_lookup_earnings(earnings_lookup, symbol)
            if entry is not None:
                hours_to = entry.hours_until(now)
                threshold = float(directives.pre_earnings_close_hours)
                if 0 <= hours_to <= threshold:
                    directive = "pre_earnings_close_hours"
                    reason = (
                        f"earnings in {hours_to:.1f}h ≤ "
                        f"{int(threshold)}h pre-earnings window"
                    )

        if reason is None or directive is None:
            continue

        seen.add(pid)
        requests.append(TradeRequest(
            instrument_id=inst_id,
            symbol=symbol,
            action="CLOSE",
            amount_usd=0.0,
            position_id=pid,
            rationale=f"directive:{directive} — {reason}",
        ))
        notes.append({
            "position_id": pid,
            "instrument_id": inst_id,
            "symbol": symbol,
            "directive": directive,
            "reason": reason,
            "asset_class": asset_class.value,
            "held_seconds": held_seconds,
        })

    return requests, notes


def _held_seconds(
    position_id: int,
    open_states: dict[int, OpenTradeState],
    *,
    now: datetime,
) -> int | None:
    state = open_states.get(int(position_id))
    if state is None:
        return None
    opened_at = getattr(state, "opened_at_iso", None)
    if not opened_at:
        return None
    try:
        dt = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=now.tzinfo)
    elapsed = (now - dt).total_seconds()
    if elapsed < 0:
        return 0
    return int(elapsed)


def _safe_lookup_earnings(lookup: Any, symbol: str) -> Any | None:
    """Run ``lookup(symbol)`` defensively.

    The earnings-calendar cache fetcher must never break a cycle. We
    swallow any unexpected exception (yfinance import error, schema
    drift, JSON corruption …) and behave as if the symbol simply has
    no scheduled earnings — the worst-case is that the bot misses a
    pre-earnings flatten on one cycle, not that the loop crashes.
    """
    try:
        return lookup(symbol)
    except Exception:  # noqa: BLE001
        return None


def _position_needs_flatten(
    *,
    meta: Any | None,
    asset_class: AssetClass,
    now: datetime,
    window_seconds: int,
) -> bool:
    """Per-instrument flatten check.

    ``no_overnight=true`` means "the bot must not hold this position
    outside its home exchange's regular session". Triggers when:

    1. The instrument's home session is currently CLOSED (weekend,
       holiday, pre-market, after-hours). The bot keeps emitting
       CLOSE requests every cycle until the broker lets us exit;
       per-instrument cooldown prevents spam on a single name.
    2. The session is open but within ``window_seconds`` of close —
       so the order lands while liquidity is still on the book.

    Crypto is never flattened (24/7). FX flattens only on weekends
    (its own "session closed").

    Unlike the previous implementation this is keyed off the
    instrument's actual exchange (LSE / XETRA / HKEX / …) — not the
    US session — so a London position doesn't get force-closed at
    NY 16:00 while LSE is in mid-session.
    """
    if asset_class == AssetClass.CRYPTO:
        return False
    state = session_for(meta, asset_class, now)
    if not state.is_open:
        return True
    window = session_window_for(meta, asset_class, now)
    if window is None:
        # FX has no fixed daily window — only the weekend close fires.
        return False
    _open_utc, close_utc = window
    delta = (close_utc - now).total_seconds()
    return 0 <= delta <= window_seconds


def _flatten_reason(
    *,
    meta: Any | None,
    asset_class: AssetClass,
    now: datetime,
    window_seconds: int,
) -> str:
    """Human-readable reason for the CLOSE — surfaced in alerts + logs."""
    label = exchange_label(meta, asset_class)
    state = session_for(meta, asset_class, now)
    if not state.is_open:
        return f"no_overnight: {label} closed — flattening position"
    window = session_window_for(meta, asset_class, now)
    if window is None:
        return f"no_overnight: {label} closing soon"
    _open_utc, close_utc = window
    delta = int((close_utc - now).total_seconds())
    return (
        f"no_overnight: {delta}s to {label} close — pre-emptive flatten"
    )


__all__ = [
    "DEFAULT_FLATTEN_WINDOW_SECONDS",
    "prescreen_candidates",
    "build_directive_close_requests",
]
