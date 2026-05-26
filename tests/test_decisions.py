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


class TuningRoundTripTests(unittest.TestCase):
    """The engine must extract an optional `tuning` block from the LLM JSON."""

    def setUp(self) -> None:
        self.guardrails = GuardrailsConfig(max_per_trade_usd=200.0)
        self.ai_cfg = AiConfig(enabled=True, veto_on_unavailable=False)

    def test_tuning_block_parsed_when_present(self) -> None:
        ai = _FakeAi(parsed={
            "actions": [
                {"instrumentId": 1, "symbol": "AAPL", "action": "HOLD"},
            ],
            "summary": "calm",
            "tuning": {
                "reason": "drought",
                "changes": [
                    {"section": "strategy", "field": "min_signal_strength",
                     "value": 0.25, "rationale": "rolling max 0.30"},
                ],
            },
        })
        engine = DecisionEngine(
            ai_cfg=self.ai_cfg, guardrails=self.guardrails, ai_client=ai,  # type: ignore[arg-type]
        )
        result = engine.decide(
            candidates=[_candidate()],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={1: "AAPL"},
        )
        self.assertFalse(result.tuning.is_empty)
        self.assertEqual(len(result.tuning.changes), 1)
        self.assertEqual(result.tuning.changes[0].field, "min_signal_strength")
        self.assertAlmostEqual(result.tuning.changes[0].value, 0.25)
        self.assertEqual(result.tuning.reason, "drought")

    def test_missing_tuning_block_returns_empty_request(self) -> None:
        ai = _FakeAi(parsed={
            "actions": [{"instrumentId": 1, "symbol": "AAPL", "action": "HOLD"}],
            "summary": "ok",
        })
        engine = DecisionEngine(
            ai_cfg=self.ai_cfg, guardrails=self.guardrails, ai_client=ai,  # type: ignore[arg-type]
        )
        result = engine.decide(
            candidates=[_candidate()],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={1: "AAPL"},
        )
        self.assertTrue(result.tuning.is_empty)

    def test_evidence_only_call_still_runs_when_no_candidates(self) -> None:
        # Even without any candidates, if there's autotune_evidence the
        # engine should call the LLM (so a stuck bot can self-unstick).
        ai = _FakeAi(parsed={
            "actions": [],
            "summary": "drought",
            "tuning": {
                "reason": "no candidates 4h",
                "changes": [
                    {"section": "strategy", "field": "min_signal_strength",
                     "value": 0.20, "rationale": "loosen"},
                ],
            },
        })
        engine = DecisionEngine(
            ai_cfg=self.ai_cfg, guardrails=self.guardrails, ai_client=ai,  # type: ignore[arg-type]
        )
        result = engine.decide(
            candidates=[],
            portfolio_summary={"equity": 10_000.0},
            bot_owned_positions=[],
            symbol_for_id={},
            autotune_evidence={"cycle_index": 100,
                               "drought": {"cycles_since_last_candidate": 240}},
        )
        self.assertTrue(result.llm_used)
        self.assertFalse(result.tuning.is_empty)

    def test_no_candidates_no_evidence_short_circuits(self) -> None:
        ai = _FakeAi(parsed={"actions": []})
        engine = DecisionEngine(
            ai_cfg=self.ai_cfg, guardrails=self.guardrails, ai_client=ai,  # type: ignore[arg-type]
        )
        result = engine.decide(
            candidates=[],
            portfolio_summary={},
            bot_owned_positions=[],
            symbol_for_id={},
        )
        self.assertEqual(ai.calls, 0)
        self.assertFalse(result.llm_used)


class RenderTests(unittest.TestCase):
    def test_render_empty(self) -> None:
        self.assertEqual(render_decisions([]), "HOLD all")


if __name__ == "__main__":
    unittest.main()
