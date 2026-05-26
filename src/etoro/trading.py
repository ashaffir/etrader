"""Trading-side wrappers: portfolio snapshot + open / close orders.

Implements the per-environment paths (``/trading/info/{env}/...`` and
``/trading/execution/{env}/...``) so the bot is agnostic to paper/live.

The portfolio response uses **capital-suffix** identifier fields per the
*etoro-account-snapshot* rule (``instrumentID``, ``positionID``, ``CID``,
``mirrorID``…) — but the example payload in the OpenAPI spec uses
lowercase variants. We accept both for resilience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .client import EtoroClient


# ---------------------------------------------------------------------------
# Portfolio snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    position_id: int
    instrument_id: int
    is_buy: bool
    open_rate: float
    amount: float
    units: float
    leverage: int
    mirror_id: int
    pnl: float
    raw: dict[str, Any]

    @property
    def is_mirror(self) -> bool:
        return bool(self.mirror_id) and self.mirror_id > 0


@dataclass(frozen=True)
class PendingOpenOrder:
    order_id: int
    instrument_id: int
    amount: float
    is_buy: bool
    leverage: int
    mirror_id: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class PortfolioSnapshot:
    credit: float
    unrealized_pnl: float
    positions: list[Position] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)         # MIT / limit orders
    orders_for_open: list[PendingOpenOrder] = field(default_factory=list)
    mirrors: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def position_by_id(self) -> dict[int, Position]:
        return {p.position_id: p for p in self.positions}

    def positions_for(self, instrument_id: int) -> list[Position]:
        return [p for p in self.positions if p.instrument_id == instrument_id]


def _g(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first matching key (handles capital-suffix vs lowerCamel)."""
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return default


def _position_from(raw: dict[str, Any]) -> Position:
    pnl = _g(raw, "pnL", "PnL", "pnl", default=None)
    if isinstance(pnl, dict):
        pnl = _g(pnl, "pnL", "pnl", default=0.0)
    return Position(
        position_id=int(_g(raw, "positionID", "positionId", default=0) or 0),
        instrument_id=int(_g(raw, "instrumentID", "instrumentId", default=0) or 0),
        is_buy=bool(_g(raw, "isBuy", default=True)),
        open_rate=float(_g(raw, "openRate", default=0.0) or 0.0),
        amount=float(_g(raw, "amount", default=0.0) or 0.0),
        units=float(_g(raw, "units", default=0.0) or 0.0),
        leverage=int(_g(raw, "leverage", default=1) or 1),
        mirror_id=int(_g(raw, "mirrorID", "mirrorId", default=0) or 0),
        pnl=float(pnl or 0.0),
        raw=raw,
    )


def _order_for_open_from(raw: dict[str, Any]) -> PendingOpenOrder:
    return PendingOpenOrder(
        order_id=int(_g(raw, "orderID", "orderId", default=0) or 0),
        instrument_id=int(_g(raw, "instrumentID", "instrumentId", default=0) or 0),
        amount=float(_g(raw, "amount", default=0.0) or 0.0),
        is_buy=bool(_g(raw, "isBuy", default=True)),
        leverage=int(_g(raw, "leverage", default=1) or 1),
        mirror_id=int(_g(raw, "mirrorID", "mirrorId", default=0) or 0),
        raw=raw,
    )


def fetch_portfolio(client: EtoroClient, env: str) -> PortfolioSnapshot:
    """Get the demo or real account snapshot.

    ``env`` is ``"demo"`` or ``"real"``.
    """
    if env not in {"demo", "real"}:
        raise ValueError(f"env must be 'demo' or 'real', got {env!r}")
    payload = client.get(f"/trading/info/{env}/pnl", retries=2) or {}
    cp = payload.get("clientPortfolio") or payload  # tolerate either shape
    positions = [_position_from(p) for p in (cp.get("positions") or [])]
    orders_for_open = [_order_for_open_from(o) for o in (cp.get("ordersForOpen") or [])]
    return PortfolioSnapshot(
        credit=float(cp.get("credit") or 0.0),
        unrealized_pnl=float(cp.get("unrealizedPnL") or 0.0),
        positions=positions,
        orders=list(cp.get("orders") or []),
        orders_for_open=orders_for_open,
        mirrors=list(cp.get("mirrors") or []),
        raw=cp,
    )


def compute_account_summary(snap: PortfolioSnapshot) -> dict[str, float]:
    """Available cash, total invested, P&L, equity — per the *etoro-account-snapshot* rule."""
    open_orders_manual = sum(
        float(_g(o.raw, "amount", default=0.0) or 0.0)
        for o in snap.orders_for_open
        if o.mirror_id == 0
    )
    open_orders_external_costs_manual = sum(
        float(_g(o.raw, "totalExternalCosts", default=0.0) or 0.0)
        for o in snap.orders_for_open
        if o.mirror_id == 0
    )
    market_orders_amount = sum(
        float(_g(o, "amount", default=0.0) or 0.0)
        for o in snap.orders
    )

    available_cash = snap.credit - open_orders_manual - market_orders_amount

    positions_invested = sum(p.amount for p in snap.positions)
    mirror_position_invested = 0.0
    mirror_avail_minus_closed = 0.0
    for m in snap.mirrors:
        for mp in m.get("positions", []) or []:
            mirror_position_invested += float(_g(mp, "amount", default=0.0) or 0.0)
        mirror_avail_minus_closed += (
            float(_g(m, "availableAmount", default=0.0) or 0.0)
            - float(_g(m, "closedPositionsNetProfit", default=0.0) or 0.0)
        )

    total_invested = (
        positions_invested
        + mirror_position_invested
        + mirror_avail_minus_closed
        + open_orders_manual
        + market_orders_amount
        + open_orders_external_costs_manual
    )

    # P&L: eToro's /pnl response is inconsistent across account
    # compositions. Sometimes ``unrealizedPnL`` at the top level is
    # populated and the per-position ``pnL`` fields are zero
    # (observed live for unleveraged cash equities); other times
    # the per-position fields carry the truth and the top-level is
    # zero. We pick whichever is non-zero; if both are zero the
    # account is genuinely flat and 0 is correct.
    pnl_positions = sum(p.pnl for p in snap.positions)
    pnl_mirrors = 0.0
    for m in snap.mirrors:
        for mp in m.get("positions", []) or []:
            inner_pnl = _g(mp, "pnL", "pnl", default=0.0)
            if isinstance(inner_pnl, dict):
                inner_pnl = _g(inner_pnl, "pnL", "pnl", default=0.0)
            pnl_mirrors += float(inner_pnl or 0.0)
        pnl_mirrors += float(_g(m, "closedPositionsNetProfit", default=0.0) or 0.0)
    per_position_pnl = pnl_positions + pnl_mirrors
    top_level_pnl = float(snap.unrealized_pnl)
    if top_level_pnl != 0.0:
        profit_loss = top_level_pnl
    else:
        profit_loss = per_position_pnl

    equity = available_cash + total_invested + profit_loss

    return {
        "credit": snap.credit,
        "available_cash": available_cash,
        "total_invested": total_invested,
        "profit_loss": profit_loss,
        "equity": equity,
    }


# ---------------------------------------------------------------------------
# Order placement / closing
# ---------------------------------------------------------------------------

def open_market_position_by_amount(
    client: EtoroClient,
    *,
    env: str,
    instrument_id: int,
    is_buy: bool,
    amount_usd: float,
    leverage: int = 1,
    stop_loss_rate: float | None = None,
    take_profit_rate: float | None = None,
) -> dict[str, Any]:
    """POST a market open by USD amount. Returns the raw response dict."""
    if env not in {"demo", "real"}:
        raise ValueError(f"env must be 'demo' or 'real', got {env!r}")
    body: dict[str, Any] = {
        "InstrumentID": int(instrument_id),
        "IsBuy": bool(is_buy),
        "Leverage": int(max(1, leverage)),
        "Amount": float(amount_usd),
    }
    if stop_loss_rate is not None:
        body["StopLossRate"] = float(stop_loss_rate)
    if take_profit_rate is not None:
        body["TakeProfitRate"] = float(take_profit_rate)
    return client.post(f"/trading/execution/{env}/market-open-orders/by-amount", json=body)


def close_position_by_market(
    client: EtoroClient,
    *,
    env: str,
    position_id: int,
    instrument_id: int,
    units_to_deduct: float | None = None,
) -> dict[str, Any]:
    """Fully close (``units_to_deduct=None``) or partially close a position."""
    if env not in {"demo", "real"}:
        raise ValueError(f"env must be 'demo' or 'real', got {env!r}")
    body: dict[str, Any] = {"InstrumentID": int(instrument_id)}
    if units_to_deduct is not None:
        body["UnitsToDeduct"] = float(units_to_deduct)
    return client.post(
        f"/trading/execution/{env}/market-close-orders/positions/{position_id}",
        json=body,
    )


def fetch_order(client: EtoroClient, env: str, order_id: int) -> dict[str, Any]:
    """Raw /trading/info/{env}/orders/{id} fetch.

    For a typed view (``OrderStatus``, error fields) prefer
    :func:`src.etoro.order_lifecycle.get_order_info`.
    """
    if env not in {"demo", "real"}:
        raise ValueError(f"env must be 'demo' or 'real', got {env!r}")
    return client.get(f"/trading/info/{env}/orders/{order_id}", retries=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_bot_owned(positions: Sequence[Position], owned_ids: set[int]) -> list[Position]:
    return [p for p in positions if p.position_id in owned_ids and p.mirror_id == 0]
