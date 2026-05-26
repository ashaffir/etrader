"""Order-lifecycle helpers: status semantics + cancel endpoints.

Carved out of :mod:`trading` to keep that module focused on portfolio +
order *placement* and to give the cancel-stuck-orders feature its own
home.

The eToro Public API exposes three pieces of order-lifecycle data we
care about:

* ``GET  /trading/info/{env}/orders/{id}``     — status, error, fills
* ``DELETE /trading/execution/{env}/market-open-orders/{id}``  — cancel
* ``DELETE /trading/execution/{env}/market-close-orders/{id}`` — cancel

The DELETE endpoints return ``{"token": "..."}`` on success and 4xx
when the order is already terminal (executed / cancelled / rejected).
Callers should expect :class:`EtoroBadRequestError` in that case and
treat it as "cancel not possible".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .client import EtoroClient


_VALID_ENVS = frozenset({"demo", "real"})


def _validate_env(env: str) -> None:
    if env not in _VALID_ENVS:
        raise ValueError(f"env must be 'demo' or 'real', got {env!r}")


def _g(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Tolerate either capital-suffix (``statusID``) or lowerCamel keys.

    Matches the pattern used by :mod:`src.etoro.trading`: eToro's docs
    show capital-suffix variants but the live API sometimes returns the
    lowerCamel form depending on the endpoint.
    """
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return default


# ---------------------------------------------------------------------------
# Status semantics
# ---------------------------------------------------------------------------

class OrderStatus(IntEnum):
    """Mirror of eToro's ``statusID`` on ``/orders/{id}`` responses.

    Values per the OpenAPI spec. Any unknown integer maps to
    :data:`UNKNOWN` so the bot fails safely (treat as still-pending
    rather than mis-reporting as filled).
    """

    PENDING = 0
    EXECUTED = 1
    CANCELLED = 2
    REJECTED = 3
    PARTIALLY_EXECUTED = 4
    UNKNOWN = -1

    @classmethod
    def from_raw(cls, value: Any) -> "OrderStatus":
        try:
            return cls(int(value))
        except (TypeError, ValueError):
            return cls.UNKNOWN

    @property
    def is_terminal(self) -> bool:
        """Order is no longer mutable (filled, cancelled, or rejected)."""
        return self in {OrderStatus.EXECUTED, OrderStatus.CANCELLED, OrderStatus.REJECTED}

    @property
    def is_pending(self) -> bool:
        """Order is still waiting in the broker's queue."""
        return self in {OrderStatus.PENDING, OrderStatus.PARTIALLY_EXECUTED, OrderStatus.UNKNOWN}


# ---------------------------------------------------------------------------
# Typed order info
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderInfo:
    """Parsed view of ``GET /trading/info/{env}/orders/{id}``.

    Used by the position monitor to (a) confirm an order actually
    executed before adopting the resulting position, and (b) re-check
    after a cancel attempt fails — to distinguish "already filled, no
    alert needed" from "genuinely stuck, alert the operator".
    """

    order_id: int
    status: OrderStatus
    error_code: int | None
    error_message: str | None
    instrument_id: int | None
    amount_usd: float | None
    raw: dict[str, Any]

    @property
    def has_error(self) -> bool:
        return self.error_code is not None or bool(self.error_message)


def get_order_info(client: EtoroClient, env: str, order_id: int) -> OrderInfo:
    """Typed wrapper around the raw fetch."""
    _validate_env(env)
    raw = client.get(f"/trading/info/{env}/orders/{order_id}", retries=2) or {}
    return _order_info_from(raw)


def _order_info_from(raw: dict[str, Any]) -> OrderInfo:
    instrument_id_raw = _g(raw, "instrumentID", "instrumentId", default=0) or 0
    amount_raw = _g(raw, "amount", default=None)
    return OrderInfo(
        order_id=int(_g(raw, "orderID", "orderId", default=0) or 0),
        status=OrderStatus.from_raw(_g(raw, "statusID", "statusId", default=None)),
        error_code=_g(raw, "errorCode", default=None),
        error_message=_g(raw, "errorMessage", default=None),
        instrument_id=int(instrument_id_raw) or None,
        amount_usd=float(amount_raw) if amount_raw is not None else None,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Cancellation endpoints
# ---------------------------------------------------------------------------

def cancel_market_open_order(
    client: EtoroClient, *, env: str, order_id: int,
) -> dict[str, Any]:
    """Cancel a pending market-open order.

    Raises :class:`~src.etoro.errors.EtoroBadRequestError` (or another
    :class:`~src.etoro.errors.EtoroApiError` subclass) when the broker
    refuses — typically because the order has already been processed.
    """
    _validate_env(env)
    return client.delete(f"/trading/execution/{env}/market-open-orders/{order_id}")


def cancel_market_close_order(
    client: EtoroClient, *, env: str, order_id: int,
) -> dict[str, Any]:
    """Cancel a pending market-close order. See :func:`cancel_market_open_order`."""
    _validate_env(env)
    return client.delete(f"/trading/execution/{env}/market-close-orders/{order_id}")
