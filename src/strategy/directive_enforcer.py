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

from ..execution.session import session_state
from ..performance.types import OpenTradeState
from .directives import Directives
from .risk import TradeRequest
from .tools.base import AssetClass, asset_class_for


# Default cushion before equity close at 21:00 UTC. When `no_overnight`
# is on and the bot is within this window of close, every non-crypto
# bot-owned position gets a CLOSE injected. Chosen to be wider than
# typical eToro market-order latency (a few seconds) but narrow enough
# to leave most of the trading day uninterrupted.
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
    * ``no_overnight``: close any non-crypto bot position if the
      US equity session is currently within
      ``flatten_window_seconds`` of close.

    A position can match both rules — we deduplicate so the cycle
    only emits one CLOSE per position.
    """
    if open_states is None:
        open_states = {}

    requests: list[TradeRequest] = []
    notes: list[dict[str, Any]] = []
    seen: set[int] = set()

    in_flatten_window = (
        directives.no_overnight
        and _is_equity_flatten_window(now=now, window_seconds=flatten_window_seconds)
    )
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
        elif in_flatten_window and asset_class != AssetClass.CRYPTO:
            directive = "no_overnight"
            reason = (
                "equity flatten window — close before US session ends"
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


# US equity regular session close in UTC — must stay in sync with
# :mod:`src.execution.session`. Centralising would be a future refactor.
_EQUITY_CLOSE_HOUR_UTC = 21
_EQUITY_CLOSE_MINUTE_UTC = 0


def _is_equity_flatten_window(*, now: datetime, window_seconds: int) -> bool:
    """True when the US equity session is currently open but within
    ``window_seconds`` of close. False on weekends / outside-session.
    """
    state = session_state(AssetClass.STOCK, now)
    if not state.is_open:
        return False
    close = now.replace(
        hour=_EQUITY_CLOSE_HOUR_UTC,
        minute=_EQUITY_CLOSE_MINUTE_UTC,
        second=0,
        microsecond=0,
    )
    delta = (close - now).total_seconds()
    return 0 <= delta <= window_seconds


__all__ = [
    "DEFAULT_FLATTEN_WINDOW_SECONDS",
    "prescreen_candidates",
    "build_directive_close_requests",
]
