"""Tests for the /signals Telegram surface and the rules-payload builder."""

import unittest

from src.config import GuardrailsConfig, StrategyConfig
from src.strategy.rules_summary import (
    ToolDescription,
    build_rules_payload,
    render_rules_text,
)
from src.telegram_service.commands import CommandContext, dispatch, parse_command


class RulesSummaryTests(unittest.TestCase):
    def test_payload_has_entry_and_exit(self) -> None:
        payload = build_rules_payload(
            strategy=StrategyConfig(),
            guardrails=GuardrailsConfig(),
            tools=[ToolDescription(
                name="sma_cross",
                family="price",
                purpose="cross",
                role="feature",
                asset_classes=("stock",),
            )],
        )
        self.assertIn("entry", payload)
        self.assertIn("exit", payload)
        self.assertEqual(len(payload["tools"]), 1)
        self.assertGreaterEqual(len(payload["pipeline"]), 5)

    def test_render_includes_pipeline_steps(self) -> None:
        payload = build_rules_payload(
            strategy=StrategyConfig(),
            guardrails=GuardrailsConfig(),
            tools=(),
        )
        text = render_rules_text(payload)
        self.assertIn("[SIGNALS]", text)
        self.assertIn("Entry (BUY)", text)
        self.assertIn("Exit (CLOSE)", text)
        self.assertIn("Pipeline:", text)


class _FakeAPI:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.payload = {
            "entry": {
                "trigger": "ALL of",
                "rules": ["SMA cross up", "RSI < 70"],
                "min_signal_strength": 0.55,
            },
            "exit": {
                "trigger": "ANY of",
                "rules": ["RSI >= 70"],
            },
            "guardrails": {"max_per_trade_usd": 500.0},
            "tools": [
                {"name": "sma_cross", "family": "price"},
                {"name": "rsi", "family": "price"},
            ],
            "tool_performance": [
                {"tool_name": "sma_cross", "observations": 30, "hits": 20, "hit_rate": 0.667},
            ],
            "pipeline": ["1. step", "2. step"],
        }

    def strategy_signals(self):
        self.calls.append("strategy_signals")
        return self.payload


class SignalsCommandTests(unittest.TestCase):
    def test_signals_command_renders_payload(self) -> None:
        import logging
        api = _FakeAPI()
        log = logging.getLogger("test.telegram.signals")
        log.addHandler(logging.NullHandler())
        ctx = CommandContext(
            api=api,  # type: ignore[arg-type]
            cmd=parse_command("/signals"),
            sender_username="alice",
            logger=log,
        )
        text = dispatch(ctx).text
        self.assertEqual(api.calls, ["strategy_signals"])
        self.assertIn("[SIGNALS]", text)
        self.assertIn("SMA cross up", text)
        self.assertIn("Tool performance", text)

    def test_rules_alias_routes_to_same_handler(self) -> None:
        import logging
        api = _FakeAPI()
        log = logging.getLogger("test.telegram.rules")
        log.addHandler(logging.NullHandler())
        ctx = CommandContext(
            api=api,  # type: ignore[arg-type]
            cmd=parse_command("/rules"),
            sender_username="bob",
            logger=log,
        )
        dispatch(ctx)
        self.assertEqual(api.calls, ["strategy_signals"])


if __name__ == "__main__":
    unittest.main()
