"""Decision engine: turn deterministic candidates + LLM overlay into TradeRequests.

If the LLM is available and enabled, we ask it to BUY/HOLD/CLOSE on each
candidate and choose a USD amount per BUY (capped downstream by the
risk layer). If the LLM is unavailable and the config has
``veto_on_unavailable = true``, we silently HOLD this cycle. Otherwise
we fall back to a deterministic mapping (BUY = max-per-trade cap; CLOSE
= go).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..ai.azure_client import AiCallResult, AzureFoundryClient, AzureUnavailable
from ..ai.prompts import build_decision_prompt
from ..config import AiConfig, GuardrailsConfig
from ..etoro.trading import Position
from .risk import TradeRequest
from .signals import Candidate
from .tools.runner import ToolRunResult


@dataclass(frozen=True)
class DecisionResult:
    requests: list[TradeRequest]
    summary: str
    llm_used: bool
    latency_ms: int | None
    raw_text: str | None


class DecisionEngine:
    """Combines deterministic candidates with optional LLM overlay."""

    def __init__(
        self,
        *,
        ai_cfg: AiConfig,
        guardrails: GuardrailsConfig,
        ai_client: AzureFoundryClient | None,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._ai_cfg = ai_cfg
        self._guardrails = guardrails
        self._ai_client = ai_client
        self._logger = logger or logging.getLogger("etrader.strategy.decisions")

    def decide(
        self,
        *,
        candidates: Sequence[Candidate],
        portfolio_summary: Mapping[str, float],
        bot_owned_positions: Sequence[Position],
        symbol_for_id: Mapping[int, str],
        market_summary: str | None = None,
        tool_results: Mapping[int, ToolRunResult] | None = None,
        cross_asset_regime: Mapping[str, Any] | None = None,
        strategy_rules: Mapping[str, Any] | None = None,
    ) -> DecisionResult:
        if not candidates:
            return DecisionResult(requests=[], summary="no candidates", llm_used=False, latency_ms=None, raw_text=None)

        tool_results = tool_results or {}

        if self._ai_cfg.enabled and self._ai_client is not None:
            try:
                ai_result = self._call_llm(
                    candidates=candidates,
                    portfolio_summary=portfolio_summary,
                    bot_owned_positions=bot_owned_positions,
                    symbol_for_id=symbol_for_id,
                    market_summary=market_summary,
                    tool_results=tool_results,
                    cross_asset_regime=cross_asset_regime,
                    strategy_rules=strategy_rules,
                )
                requests = self._requests_from_llm(ai_result.parsed_json, candidates, bot_owned_positions)
                return DecisionResult(
                    requests=requests,
                    summary=self._summary_from_llm(ai_result.parsed_json),
                    llm_used=True,
                    latency_ms=ai_result.latency_ms,
                    raw_text=ai_result.text,
                )
            except AzureUnavailable as exc:
                self._logger.warning("LLM unavailable, %s", "vetoing trades" if self._ai_cfg.veto_on_unavailable else "falling back deterministic")
                if self._ai_cfg.veto_on_unavailable:
                    return DecisionResult(
                        requests=[],
                        summary=f"LLM unavailable: {exc}; veto active",
                        llm_used=False,
                        latency_ms=None,
                        raw_text=None,
                    )

        # Deterministic fallback. Honors hard gates from tool_results.
        return DecisionResult(
            requests=self._deterministic_requests(candidates, bot_owned_positions, tool_results),
            summary="deterministic fallback (LLM disabled or unavailable)",
            llm_used=False,
            latency_ms=None,
            raw_text=None,
        )

    # ------------------------------------------------------------------

    def _call_llm(
        self,
        *,
        candidates: Sequence[Candidate],
        portfolio_summary: Mapping[str, float],
        bot_owned_positions: Sequence[Position],
        symbol_for_id: Mapping[int, str],
        market_summary: str | None,
        tool_results: Mapping[int, ToolRunResult],
        cross_asset_regime: Mapping[str, Any] | None,
        strategy_rules: Mapping[str, Any] | None,
    ) -> AiCallResult:
        assert self._ai_client is not None
        owned = [
            {
                "instrumentId": p.instrument_id,
                "symbol": symbol_for_id.get(p.instrument_id, str(p.instrument_id)),
                "positionId": p.position_id,
                "amount": p.amount,
                "openRate": p.open_rate,
                "pnl": p.pnl,
                "isBuy": p.is_buy,
            }
            for p in bot_owned_positions
        ]
        cands = [
            self._candidate_to_dict(c, tool_results.get(c.instrument_id))
            for c in candidates
        ]
        guardrails_summary = {
            "max_per_trade_usd": self._guardrails.max_per_trade_usd,
            "max_parallel_trades": self._guardrails.max_parallel_trades,
            "default_stop_loss_pct": self._guardrails.default_stop_loss_pct,
            "default_take_profit_pct": self._guardrails.default_take_profit_pct,
            "max_leverage": self._guardrails.max_leverage,
        }
        system, user = build_decision_prompt(
            portfolio_summary=portfolio_summary,
            bot_owned_positions=owned,
            candidates=cands,
            guardrails_summary=guardrails_summary,
            market_summary=market_summary,
            cross_asset_regime=cross_asset_regime,
            strategy_rules=strategy_rules,
        )
        return self._ai_client.chat_json(system=system, user=user, require_json=True)

    def _requests_from_llm(
        self,
        parsed: Any,
        candidates: Sequence[Candidate],
        bot_owned_positions: Sequence[Position],
    ) -> list[TradeRequest]:
        if not parsed or not isinstance(parsed, dict):
            self._logger.warning("LLM returned no parsable JSON; falling back deterministic")
            return self._deterministic_requests(candidates, bot_owned_positions)
        actions = parsed.get("actions") or []
        if not isinstance(actions, list):
            return self._deterministic_requests(candidates, bot_owned_positions)
        cand_by_inst = {c.instrument_id: c for c in candidates}
        owned_by_inst: dict[int, Position] = {p.instrument_id: p for p in bot_owned_positions}
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
            if action == "BUY":
                amount = self._safe_float(entry.get("amount_usd"), default=0.0)
                if amount <= 0:
                    amount = float(self._guardrails.max_per_trade_usd)
                amount = min(amount, float(self._guardrails.max_per_trade_usd))
                out.append(TradeRequest(
                    instrument_id=inst_id, symbol=symbol, action="BUY",
                    amount_usd=amount, position_id=None,
                ))
            elif action == "CLOSE":
                pos = owned_by_inst.get(inst_id)
                if pos is None:
                    continue
                out.append(TradeRequest(
                    instrument_id=inst_id, symbol=symbol, action="CLOSE",
                    amount_usd=0.0, position_id=pos.position_id,
                ))
        return out

    def _deterministic_requests(
        self,
        candidates: Sequence[Candidate],
        bot_owned_positions: Sequence[Position],
        tool_results: Mapping[int, ToolRunResult] | None = None,
    ) -> list[TradeRequest]:
        """Deterministic fallback. Honors hard gates from ``tool_results``.

        A BUY candidate whose tool stack vetoed it (gate_passed = False)
        never becomes a TradeRequest, even when the LLM is offline.
        """
        owned_by_inst = {p.instrument_id: p for p in bot_owned_positions}
        results = tool_results or {}
        out: list[TradeRequest] = []
        for c in candidates:
            tr = results.get(c.instrument_id)
            if c.action == "BUY":
                if tr is not None and not tr.gate_passed:
                    self._logger.info(
                        "[decisions] vetoed BUY %s (deterministic, gate=%s)",
                        c.symbol, tr.gate_reason or "?",
                    )
                    continue
                out.append(TradeRequest(
                    instrument_id=c.instrument_id, symbol=c.symbol, action="BUY",
                    amount_usd=float(self._guardrails.max_per_trade_usd), position_id=None,
                ))
            elif c.action == "CLOSE":
                pos = owned_by_inst.get(c.instrument_id)
                if pos is None:
                    continue
                out.append(TradeRequest(
                    instrument_id=c.instrument_id, symbol=c.symbol, action="CLOSE",
                    amount_usd=0.0, position_id=pos.position_id,
                ))
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_to_dict(
        c: Candidate,
        tool_result: ToolRunResult | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "instrumentId": c.instrument_id,
            "symbol": c.symbol,
            "action_hint": c.action,
            "strength": c.strength,
            "raw_score": c.raw_score,
            "ensemble": [
                {"name": comp.name, "score": comp.score, "detail": comp.detail}
                for comp in c.components
            ],
            "reason": c.reason,
            "last_close": c.last_close,
            "rsi": c.rsi,
            "sma_short": c.sma_short,
            "sma_long": c.sma_long,
            "momentum_pct": c.momentum_pct,
        }
        if tool_result is not None:
            out["tools"] = {
                "selected": list(tool_result.selected_tools),
                "features": dict(tool_result.features),
                "scores": dict(tool_result.scores),
                "aggregate_score": tool_result.aggregate_score,
                "gate_passed": tool_result.gate_passed,
                "gate_reason": tool_result.gate_reason,
            }
        return out

    @staticmethod
    def _summary_from_llm(parsed: Any) -> str:
        if isinstance(parsed, dict):
            return str(parsed.get("summary") or "").strip()
        return ""

    @staticmethod
    def _safe_float(v: Any, *, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default


def render_decisions(requests: Iterable[TradeRequest]) -> str:
    parts: list[str] = []
    for r in requests:
        if r.action == "BUY":
            parts.append(f"BUY {r.symbol} ${r.amount_usd:.2f}")
        elif r.action == "CLOSE":
            parts.append(f"CLOSE {r.symbol}")
    return "; ".join(parts) if parts else "HOLD all"
