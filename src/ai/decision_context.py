"""Project tracker + position-review state into decision-prompt payloads.

Pure helpers — no I/O. They take broker positions + tracker state +
review annotations and produce the dicts that
:func:`src.ai.prompts.build_decision_prompt` consumes.

Kept separate from ``prompts.py`` so the prompt module stays focused
on string templates and the projection logic is independently
testable.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..execution.exchange_session import resolve_exchange_label_for


def enrich_owned_position(
    *,
    position: Any,
    symbol: str,
    open_state: Any | None,
    dynamic_band: Any | None,
    review: Any | None,
    meta: Any | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Return a single bot-owned-position dict for the LLM payload.

    Always includes broker-side fields. Augments with tracker fields
    (MFE/MAE/time held / pct) when ``open_state`` is provided, with
    the active dynamic SL/TP band when ``dynamic_band`` is provided,
    the review annotation (triggers + notes) when ``review`` is
    provided, and the resolved exchange when ``meta`` is provided
    (eToro :class:`InstrumentMeta` carrying ``price_source``).
    """
    now = float(now_epoch if now_epoch is not None else time.time())
    out: dict[str, Any] = {
        "instrumentId": int(position.instrument_id),
        "symbol": symbol,
        "positionId": int(position.position_id),
        "amount_usd": float(position.amount or 0.0),
        "open_rate": float(position.open_rate or 0.0),
        "pnl_usd": float(position.pnl or 0.0),
        "is_buy": bool(position.is_buy),
        "exchange": resolve_exchange_label_for(meta, symbol),
    }
    if open_state is not None:
        opened_iso = str(getattr(open_state, "opened_at_iso", "") or "")
        opened_epoch = _iso_to_epoch(opened_iso, fallback=now)
        time_held = int(max(0.0, now - opened_epoch))
        out.update(
            {
                "pnl_pct": _round_or_none(
                    getattr(open_state, "last_pnl_pct", None), 4
                ),
                "mfe_usd": round(float(getattr(open_state, "mfe_usd", 0.0)), 4),
                "mae_usd": round(float(getattr(open_state, "mae_usd", 0.0)), 4),
                "time_held_seconds": time_held,
                "time_held_minutes": round(time_held / 60.0, 1),
                "opened_at": opened_iso or None,
                "asset_class": str(getattr(open_state, "asset_class", "") or ""),
                "snapshots": int(getattr(open_state, "snapshots", 0) or 0),
            }
        )
    if dynamic_band is not None:
        out["stops"] = {
            "stop_loss_pct": float(getattr(dynamic_band, "stop_loss_pct", 0.0)),
            "take_profit_pct": float(getattr(dynamic_band, "take_profit_pct", 0.0)),
            "trailing_stop_pct": _opt_float(
                getattr(dynamic_band, "trailing_stop_pct", None)
            ),
            "mfe_pct_observed": float(getattr(dynamic_band, "mfe_pct", 0.0)),
            "rationale": str(getattr(dynamic_band, "rationale", "") or ""),
        }
    if review is not None:
        triggers = list(getattr(review, "triggers", None) or [])
        notes = list(getattr(review, "notes", None) or [])
        out["review"] = {
            "triggers": triggers,
            "notes": notes,
        }
    return out


def project_bot_owned_positions(
    *,
    positions: Sequence[Any],
    symbol_for_id: Mapping[int, str],
    open_states: Mapping[int, Any] | None = None,
    dynamic_stops: Any | None = None,
    reviews_by_position_id: Mapping[int, Any] | None = None,
    instrument_metas: Mapping[int, Any] | None = None,
    now_epoch: float | None = None,
) -> list[dict[str, Any]]:
    """Build the ``bot_owned_positions`` block for the decision prompt.

    ``instrument_metas`` (optional) is the cycle's ``ctx.instrument_metas``
    map — when supplied every position carries an ``exchange`` label so
    the LLM can reason about session timing per market.
    """
    open_map = dict(open_states or {})
    review_map = dict(reviews_by_position_id or {})
    metas = dict(instrument_metas or {})
    out: list[dict[str, Any]] = []
    for pos in positions:
        sym = symbol_for_id.get(pos.instrument_id, f"INST-{pos.instrument_id}")
        band = (
            dynamic_stops.effective_band(pos.position_id)
            if dynamic_stops is not None and dynamic_stops.has_override(pos.position_id)
            else None
        )
        out.append(
            enrich_owned_position(
                position=pos,
                symbol=sym,
                open_state=open_map.get(pos.position_id),
                dynamic_band=band,
                review=review_map.get(pos.position_id),
                meta=metas.get(pos.instrument_id),
                now_epoch=now_epoch,
            )
        )
    return out


def project_by_symbol_history(
    *,
    by_symbol: Sequence[Mapping[str, Any]],
    symbols_of_interest: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Project the tracker's by-symbol roll-up keyed by symbol.

    ``symbols_of_interest`` is the union of the candidate symbols and
    the currently-owned symbols — anything outside it is dropped to
    keep the prompt small.
    """
    wanted = {s.upper() for s in symbols_of_interest if s}
    out: dict[str, dict[str, Any]] = {}
    for row in by_symbol:
        sym = str(row.get("symbol") or "").upper()
        if not sym or (wanted and sym not in wanted):
            continue
        out[sym] = {
            "trades": int(row.get("trades") or 0),
            "wins": int(row.get("wins") or 0),
            "losses": int(row.get("losses") or 0),
            "realized_pnl_usd": round(float(row.get("realized_pnl_usd") or 0.0), 2),
            "win_rate": round(float(row.get("win_rate") or 0.0), 3),
            "avg_pnl_usd": round(float(row.get("avg_pnl_usd") or 0.0), 2),
            "avg_hold_minutes": round(
                float(row.get("avg_hold_seconds") or 0.0) / 60.0, 1
            ),
        }
    return out


def build_performance_block(
    *,
    summary: Mapping[str, Any] | None,
    reviews: Sequence[Any] | None = None,
    by_symbol_projection: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Assemble the ``performance`` payload for the decision prompt.

    Returns ``None`` when there is nothing meaningful to report (no
    summary, no reviews, no per-symbol history).
    """
    if not summary and not reviews and not by_symbol_projection:
        return None
    block: dict[str, Any] = {}
    if summary:
        # We strip ``open`` from the bot summary block — the per-position
        # detail already lives in ``bot_owned_positions``.
        bot = dict(summary.get("bot") or {})
        account = dict(summary.get("account") or {})
        by_period = dict(summary.get("by_period") or {})
        block["bot"] = bot
        block["account"] = account
        block["by_period"] = by_period
    if reviews:
        block["position_reviews"] = [
            r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in reviews
        ]
    if by_symbol_projection:
        block["by_symbol"] = dict(by_symbol_projection)
    return block


def _iso_to_epoch(iso: str, *, fallback: float) -> float:
    if not iso:
        return fallback
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.timestamp()
    except (TypeError, ValueError):
        return fallback


def _round_or_none(v: Any, ndigits: int) -> float | None:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
