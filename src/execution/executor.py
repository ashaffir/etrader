"""Order executor.

Turns risk-approved :class:`TradeVerdict`s into HTTP calls to the eToro
trading endpoints. Respects:

- The 20 req/min trade-execution rate limit (we sleep
  ``operations.trade_spacing_seconds`` between calls).
- The "no idempotency key" rule: trade-execution POSTs are not retried
  on timeout / 5xx; we tag the verdict as ``ambiguous`` so the
  reconciliation step can verify by re-reading the portfolio.

Paper mode (``trading=paper`` in config) sends to the demo endpoints —
real money is never touched. The ``[mode] trading="live"`` path is
gated by ``ALLOW_REAL=true`` and a real user key in ``.env`` (enforced
at startup by :mod:`src.config`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

from ..config import GuardrailsConfig, OperationsConfig
from ..etoro.client import EtoroClient
from ..etoro.errors import (
    EtoroApiError,
    EtoroAuthError,
    EtoroRateLimitError,
    EtoroServerError,
    EtoroTimeoutError,
)
from ..etoro.market_data import LiveRate
from ..etoro.trading import (
    close_position_by_market,
    open_market_position_by_amount,
)
from ..state import BotState
from ..strategy.risk import TradeVerdict, compute_stop_loss_take_profit


@dataclass(frozen=True)
class ExecutionResult:
    request_symbol: str
    action: str           # "BUY" | "CLOSE"
    status: str           # "ok" | "failed" | "ambiguous" | "skipped" | "rate_limited"
    order_id: int | None = None
    position_id: int | None = None
    instrument_id: int | None = None
    amount_usd: float | None = None
    detail: str = ""


class TradeExecutor:
    def __init__(
        self,
        *,
        client: EtoroClient,
        env: str,                    # "demo" | "real"
        guardrails: GuardrailsConfig,
        operations: OperationsConfig,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        if env not in {"demo", "real"}:
            raise ValueError(f"env must be 'demo' or 'real', got {env!r}")
        self._client = client
        self._env = env
        self._guardrails = guardrails
        self._operations = operations
        self._logger = logger or logging.getLogger("etrader.execution.executor")

    def execute_all(
        self,
        *,
        verdicts: Sequence[TradeVerdict],
        rates: dict[int, LiveRate],
        state: BotState,
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        first = True
        for v in verdicts:
            if not v.approved:
                results.append(ExecutionResult(
                    request_symbol=v.request.symbol,
                    action=v.request.action,
                    status="skipped",
                    instrument_id=v.request.instrument_id,
                    detail=v.reason,
                ))
                self._logger.info("[skipped] %s %s — %s",
                                  v.request.action, v.request.symbol, v.reason)
                continue
            if not first:
                time.sleep(self._operations.trade_spacing_seconds)
            first = False
            if v.request.action == "BUY":
                results.append(self._open(v, rates, state))
            elif v.request.action == "CLOSE":
                results.append(self._close(v, state))
        return results

    # ------------------------------------------------------------------

    def _open(
        self,
        verdict: TradeVerdict,
        rates: dict[int, LiveRate],
        state: BotState,
    ) -> ExecutionResult:
        req = verdict.request
        amount = verdict.amended_amount_usd or req.amount_usd
        rate = rates.get(req.instrument_id)
        if rate is None or rate.ask is None or rate.bid is None:
            self._logger.warning("[exec] cannot OPEN %s — no live rate", req.symbol)
            return ExecutionResult(
                request_symbol=req.symbol, action="BUY", status="failed",
                instrument_id=req.instrument_id, detail="no live rate",
            )
        is_buy = True  # bot only places long buys; SHORT is out-of-scope by design
        entry_price = rate.ask if is_buy else rate.bid
        sl, tp = compute_stop_loss_take_profit(
            entry_price=entry_price,
            is_buy=is_buy,
            stop_loss_pct=self._guardrails.default_stop_loss_pct,
            take_profit_pct=self._guardrails.default_take_profit_pct,
        )

        self._logger.info(
            "[exec] OPEN %s %.2f USD long  SL=%.4f  TP=%.4f  (mid=%.4f)",
            req.symbol, amount, sl, tp, entry_price,
        )

        try:
            response = open_market_position_by_amount(
                self._client,
                env=self._env,
                instrument_id=req.instrument_id,
                is_buy=is_buy,
                amount_usd=amount,
                leverage=int(self._guardrails.max_leverage),
                stop_loss_rate=sl,
                take_profit_rate=tp,
            )
        except EtoroAuthError as exc:
            self._logger.error("[exec] AUTH error opening %s: %s", req.symbol, exc)
            return ExecutionResult(req.symbol, "BUY", "failed", instrument_id=req.instrument_id, detail=str(exc))
        except EtoroRateLimitError as exc:
            self._logger.warning("[exec] rate-limited opening %s: %s", req.symbol, exc)
            return ExecutionResult(req.symbol, "BUY", "rate_limited", instrument_id=req.instrument_id, detail=str(exc))
        except (EtoroTimeoutError, EtoroServerError) as exc:
            # Ambiguous: server may or may not have placed the order. Reconcile later.
            self._logger.warning("[exec] AMBIGUOUS opening %s: %s — reconcile via /pnl", req.symbol, exc)
            return ExecutionResult(req.symbol, "BUY", "ambiguous", instrument_id=req.instrument_id, detail=str(exc))
        except EtoroApiError as exc:
            self._logger.error("[exec] failed opening %s: %s", req.symbol, exc)
            return ExecutionResult(req.symbol, "BUY", "failed", instrument_id=req.instrument_id, detail=str(exc))

        order_for_open = (response or {}).get("orderForOpen") or {}
        order_id = order_for_open.get("orderID") or order_for_open.get("orderId")
        state.mark_action(req.instrument_id)
        state.record_bot_action()
        return ExecutionResult(
            request_symbol=req.symbol,
            action="BUY",
            status="ok",
            order_id=int(order_id) if order_id else None,
            instrument_id=req.instrument_id,
            amount_usd=amount,
            detail=f"orderID={order_id}",
        )

    def _close(self, verdict: TradeVerdict, state: BotState) -> ExecutionResult:
        req = verdict.request
        if req.position_id is None:
            return ExecutionResult(req.symbol, "CLOSE", "failed", detail="missing position_id")

        self._logger.info("[exec] CLOSE %s positionID=%d", req.symbol, req.position_id)

        try:
            response = close_position_by_market(
                self._client,
                env=self._env,
                position_id=req.position_id,
                instrument_id=req.instrument_id,
            )
        except EtoroAuthError as exc:
            self._logger.error("[exec] AUTH error closing %s: %s", req.symbol, exc)
            return ExecutionResult(req.symbol, "CLOSE", "failed", position_id=req.position_id, detail=str(exc))
        except EtoroRateLimitError as exc:
            self._logger.warning("[exec] rate-limited closing %s: %s", req.symbol, exc)
            return ExecutionResult(req.symbol, "CLOSE", "rate_limited", position_id=req.position_id, detail=str(exc))
        except (EtoroTimeoutError, EtoroServerError) as exc:
            self._logger.warning("[exec] AMBIGUOUS closing %s: %s", req.symbol, exc)
            return ExecutionResult(req.symbol, "CLOSE", "ambiguous", position_id=req.position_id, detail=str(exc))
        except EtoroApiError as exc:
            self._logger.error("[exec] failed closing %s: %s", req.symbol, exc)
            return ExecutionResult(req.symbol, "CLOSE", "failed", position_id=req.position_id, detail=str(exc))

        order_for_close = (response or {}).get("orderForClose") or {}
        order_id = order_for_close.get("orderID") or order_for_close.get("orderId")
        state.mark_action(req.instrument_id)
        state.record_bot_action()
        state.remove_owned(req.position_id)
        return ExecutionResult(
            request_symbol=req.symbol,
            action="CLOSE",
            status="ok",
            order_id=int(order_id) if order_id else None,
            position_id=req.position_id,
            instrument_id=req.instrument_id,
            detail=f"orderID={order_id}",
        )
