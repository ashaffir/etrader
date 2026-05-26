"""Parse the LLM decision JSON into :class:`TradeRequest` objects.

Extracted from :mod:`src.strategy.decisions` so the engine file stays
under the line cap and the parsing logic is independently testable
(no AI client / config dependency).

The parser is intentionally permissive: it silently drops malformed
entries instead of failing the whole cycle. The engine logs a warning
and falls back to deterministic candidates if the LLM returns no
parsable JSON at all.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from ..config import GuardrailsConfig
from ..etoro.trading import Position
from .risk import TradeRequest
from .signals import Candidate


def parse_actions(
    parsed: Any,
    *,
    candidates: Sequence[Candidate],
    bot_owned_positions: Sequence[Position],
    guardrails: GuardrailsConfig,
    position_units_by_id: Mapping[int, float] | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> list[TradeRequest] | None:
    """Parse the LLM JSON response into trade requests.

    Returns ``None`` when the LLM produced no parsable JSON (caller
    should fall back to deterministic). Returns an empty list when
    the JSON parsed cleanly but contained no actionable items.
    """
    log = logger or logging.getLogger("etrader.strategy.decision_parser")
    if not parsed or not isinstance(parsed, dict):
        log.warning("LLM returned no parsable JSON; falling back deterministic")
        return None
    actions = parsed.get("actions") or []
    if not isinstance(actions, list):
        return None
    cand_by_inst = {c.instrument_id: c for c in candidates}
    owned_by_inst: dict[int, Position] = {p.instrument_id: p for p in bot_owned_positions}
    owned_by_pos: dict[int, Position] = {p.position_id: p for p in bot_owned_positions}
    units_lookup = dict(position_units_by_id or {})
    out: list[TradeRequest] = []
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action", "HOLD")).upper()
        if action == "HOLD":
            continue
        try:
            inst_id = int(entry.get("instrumentId") or 0)
        except (TypeError, ValueError):
            continue
        cand = cand_by_inst.get(inst_id)
        symbol = (cand.symbol if cand else str(entry.get("symbol", inst_id))).upper()
        rationale = str(entry.get("rationale") or "")
        if action == "BUY":
            req = _build_buy(entry, inst_id, symbol, rationale, guardrails)
        elif action == "CLOSE":
            req = _build_close(
                entry, inst_id, symbol, rationale,
                owned_by_inst=owned_by_inst,
                owned_by_pos=owned_by_pos,
                units_lookup=units_lookup,
            )
        elif action == "MODIFY_STOPS":
            req = _build_modify_stops(
                entry, inst_id, symbol, rationale,
                owned_by_inst=owned_by_inst,
                owned_by_pos=owned_by_pos,
            )
        else:
            req = None
        if req is not None:
            out.append(req)
    return out


def _build_buy(
    entry: Mapping[str, Any],
    inst_id: int,
    symbol: str,
    rationale: str,
    guardrails: GuardrailsConfig,
) -> TradeRequest | None:
    amount = _safe_float(entry.get("amount_usd"), default=0.0)
    if amount is None or amount <= 0:
        amount = float(guardrails.max_per_trade_usd)
    amount = min(amount, float(guardrails.max_per_trade_usd))
    return TradeRequest(
        instrument_id=inst_id, symbol=symbol, action="BUY",
        amount_usd=amount, position_id=None, rationale=rationale,
    )


def _build_close(
    entry: Mapping[str, Any],
    inst_id: int,
    symbol: str,
    rationale: str,
    *,
    owned_by_inst: Mapping[int, Position],
    owned_by_pos: Mapping[int, Position],
    units_lookup: Mapping[int, float],
) -> TradeRequest | None:
    pos = _resolve_owned_position(entry, inst_id, owned_by_inst, owned_by_pos)
    if pos is None:
        return None
    fraction = _safe_float(entry.get("close_fraction"), default=None)
    close_units: float | None = None
    if fraction is not None and 0.0 < float(fraction) < 1.0:
        base_units = float(units_lookup.get(pos.position_id, 0.0) or 0.0)
        if base_units > 0.0:
            close_units = base_units * float(fraction)
    return TradeRequest(
        instrument_id=inst_id, symbol=symbol, action="CLOSE",
        amount_usd=0.0, position_id=pos.position_id,
        close_fraction=fraction, close_units=close_units,
        rationale=rationale,
    )


def _build_modify_stops(
    entry: Mapping[str, Any],
    inst_id: int,
    symbol: str,
    rationale: str,
    *,
    owned_by_inst: Mapping[int, Position],
    owned_by_pos: Mapping[int, Position],
) -> TradeRequest | None:
    pos = _resolve_owned_position(entry, inst_id, owned_by_inst, owned_by_pos)
    if pos is None:
        return None
    sl = _safe_float(entry.get("stop_loss_pct"), default=None)
    tp = _safe_float(entry.get("take_profit_pct"), default=None)
    trail = _safe_float(entry.get("trailing_stop_pct"), default=None)
    if sl is None and tp is None and trail is None:
        return None
    return TradeRequest(
        instrument_id=inst_id, symbol=symbol, action="MODIFY_STOPS",
        amount_usd=0.0, position_id=pos.position_id,
        stop_loss_pct=sl, take_profit_pct=tp, trailing_stop_pct=trail,
        rationale=rationale,
    )


def _resolve_owned_position(
    entry: Mapping[str, Any],
    inst_id: int,
    owned_by_inst: Mapping[int, Position],
    owned_by_pos: Mapping[int, Position],
) -> Position | None:
    """Find the bot-owned position the action targets."""
    try:
        pos_id = int(entry.get("positionId") or entry.get("position_id") or 0)
    except (TypeError, ValueError):
        pos_id = 0
    if pos_id > 0 and pos_id in owned_by_pos:
        return owned_by_pos[pos_id]
    return owned_by_inst.get(inst_id)


def _safe_float(v: Any, *, default: float | None = 0.0) -> float | None:
    """Coerce ``v`` to ``float``; return ``default`` on failure or None input."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
