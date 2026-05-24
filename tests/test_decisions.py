"""Tests for the decision engine — covers both the LLM and deterministic paths.

We feed a synthetic ``Candidate`` list and inject a fake LLM client to
verify the JSON contract the production prompt expects.
"""

import unittest
from typing import Any

from src.ai.azure_client import AiCallResult, AzureUnavailable
from src.config import AiConfig, GuardrailsConfig
from src.etoro.trading import Position
from src.strategy.decisions import DecisionEngine, render_decisions
from src.strategy.signals import Candidate


class _FakeAi:
    def __init__(self, parsed: Any | None = None, raise_unavailable: bool = False) -> None:
        self.parsed = parsed
        self.raise_unavailable = raise_unavailable
        self.calls = 0

    def chat_json(self, *, system: str, user: str, require_json: bool = True) -> AiCallResult:  # noqa: ARG002
        self.calls += 1
        if self.raise_unavailable:
            raise AzureUnavailable("test-down")
        return AiCallResult(text="{}", parsed_json=self.parsed, latency_ms=42)


def _candidate(action: str = "BUY", inst_id: int = 1, symbol: str = "AAPL", strength: float = 0.8) -> Candidate:
    return Candidate(
        instrument_id=inst_id, symbol=symbol, action=action, strength=strength,
        reason="test", last_close=100.0, rsi=55.0, sma_short=101.0, sma_long=99.0,
        momentum_pct=2.0,
    )


def _position(position_id: int, instrument_id: int) -> Position:
    return Position(
        position_id=position_id, instrument_id=instrument_id, is_buy=True,
        open_rate=100.0, amount=300.0, units=3.0, leverage=1, mirror_id=0,
        pnl=10.0, raw={},
    )


class DecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guardrails = GuardrailsConfig(max_per_trade_usd=200.0, max_parallel_trades=10)
        self.ai_cfg = AiConfig(enabled=True, veto_on_unavailable=False)

    def test_llm_produces_buy_request(self) -> None:
        ai = _FakeAi(parsed={
            "actions": [
                {"instrumentId": 1, "symbol": "AAPL", "action": "BUY",
                 "amount_usd": 150.0, "confidence": 0.7, "rationale": "x"},
            ],
            "summary": "ok",
        })
        engine = DecisionEngine(ai_cfg=self.ai_cfg, guardrails=self.guardrails, ai_client=ai)  # type: ignore[arg-type]
        result = engine.decide(
            candidates=[_candidate()],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={1: "AAPL"},
        )
        self.assertTrue(result.llm_used)
        self.assertEqual(len(result.requests), 1)
        self.assertEqual(result.requests[0].action, "BUY")
        self.assertAlmostEqual(result.requests[0].amount_usd, 150.0)

    def test_llm_buy_amount_capped_to_per_trade(self) -> None:
        ai = _FakeAi(parsed={
            "actions": [
                {"instrumentId": 1, "symbol": "AAPL", "action": "BUY",
                 "amount_usd": 5_000.0, "confidence": 0.9},
            ],
        })
        engine = DecisionEngine(ai_cfg=self.ai_cfg, guardrails=self.guardrails, ai_client=ai)  # type: ignore[arg-type]
        result = engine.decide(
            candidates=[_candidate()],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={1: "AAPL"},
        )
        self.assertAlmostEqual(result.requests[0].amount_usd, 200.0)  # capped

    def test_llm_close_requires_owned_position(self) -> None:
        ai = _FakeAi(parsed={
            "actions": [
                {"instrumentId": 1, "symbol": "AAPL", "action": "CLOSE"},
            ],
        })
        engine = DecisionEngine(ai_cfg=self.ai_cfg, guardrails=self.guardrails, ai_client=ai)  # type: ignore[arg-type]
        # No position owned → CLOSE silently dropped
        result = engine.decide(
            candidates=[_candidate(action="CLOSE")],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={1: "AAPL"},
        )
        self.assertEqual(result.requests, [])

    def test_unavailable_with_veto_returns_empty(self) -> None:
        ai = _FakeAi(raise_unavailable=True)
        engine = DecisionEngine(
            ai_cfg=AiConfig(enabled=True, veto_on_unavailable=True),
            guardrails=self.guardrails,
            ai_client=ai,  # type: ignore[arg-type]
        )
        result = engine.decide(
            candidates=[_candidate()],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={1: "AAPL"},
        )
        self.assertFalse(result.llm_used)
        self.assertEqual(result.requests, [])
        self.assertIn("veto", result.summary)

    def test_unavailable_without_veto_uses_deterministic(self) -> None:
        ai = _FakeAi(raise_unavailable=True)
        engine = DecisionEngine(
            ai_cfg=AiConfig(enabled=True, veto_on_unavailable=False),
            guardrails=self.guardrails,
            ai_client=ai,  # type: ignore[arg-type]
        )
        result = engine.decide(
            candidates=[_candidate()],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={1: "AAPL"},
        )
        self.assertFalse(result.llm_used)
        self.assertEqual(len(result.requests), 1)
        self.assertEqual(result.requests[0].action, "BUY")
        self.assertAlmostEqual(result.requests[0].amount_usd, 200.0)

    def test_disabled_ai_uses_deterministic(self) -> None:
        engine = DecisionEngine(
            ai_cfg=AiConfig(enabled=False),
            guardrails=self.guardrails,
            ai_client=None,
        )
        result = engine.decide(
            candidates=[_candidate(action="CLOSE")],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[_position(position_id=42, instrument_id=1)],
            symbol_for_id={1: "AAPL"},
        )
        self.assertEqual(len(result.requests), 1)
        self.assertEqual(result.requests[0].action, "CLOSE")
        self.assertEqual(result.requests[0].position_id, 42)


class RenderTests(unittest.TestCase):
    def test_render_empty(self) -> None:
        self.assertEqual(render_decisions([]), "HOLD all")


if __name__ == "__main__":
    unittest.main()
