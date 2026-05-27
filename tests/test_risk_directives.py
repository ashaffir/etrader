"""Tests for the directives integration into RiskEvaluator."""

from __future__ import annotations

import unittest

from src.config import GuardrailsConfig
from src.state import BotState
from src.strategy.directives import Directives
from src.strategy.risk import RiskEvaluator, TradeRequest


def _eval(directives: Directives | None = None, **overrides) -> RiskEvaluator:
    cfg = GuardrailsConfig(
        max_per_trade_usd=overrides.get("max_per_trade_usd", 500.0),
        max_parallel_trades=overrides.get("max_parallel_trades", 10),
        daily_loss_stop_usd=overrides.get("daily_loss_stop_usd", 250.0),
        per_instrument_cooldown_min=overrides.get("per_instrument_cooldown_min", 60),
        default_stop_loss_pct=5.0,
        default_take_profit_pct=8.0,
        max_leverage=1,
        max_bot_invested_usd=overrides.get("max_bot_invested_usd", 0.0),
        min_amend_remainder_usd=overrides.get("min_amend_remainder_usd", 50.0),
    )
    if directives is None:
        return RiskEvaluator(cfg)
    return RiskEvaluator(cfg, directives_provider=lambda: directives)


class BlockedSymbolTests(unittest.TestCase):
    def test_buy_for_blocked_symbol_is_rejected(self) -> None:
        ev = _eval(directives=Directives(blocked_symbols=("NVDA",)))
        verdicts = ev.evaluate(
            requests=[TradeRequest(99, "NVDA", "BUY", 100.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertFalse(verdicts[0].approved)
        self.assertIn("blocked_symbols", verdicts[0].reason)
        self.assertIn("NVDA", verdicts[0].reason)

    def test_buy_for_other_symbol_passes(self) -> None:
        ev = _eval(directives=Directives(blocked_symbols=("NVDA",)))
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 100.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertTrue(verdicts[0].approved)

    def test_close_for_blocked_symbol_still_allowed(self) -> None:
        ev = _eval(directives=Directives(blocked_symbols=("NVDA",)))
        state = BotState()
        state.bot_owned_positions.add(42)
        verdicts = ev.evaluate(
            requests=[TradeRequest(99, "NVDA", "CLOSE", 0.0, position_id=42)],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=1,
        )
        # CLOSE never hits the symbol-block guard — only BUYs do.
        self.assertTrue(verdicts[0].approved)


class AccountTotalCapTests(unittest.TestCase):
    def test_cap_zero_means_disabled(self) -> None:
        ev = _eval(directives=Directives(max_total_account_invested_usd=0.0))
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 500.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
            account_invested_total_usd=99_999.0,
        )
        self.assertTrue(verdicts[0].approved)

    def test_buy_within_headroom_is_approved_unchanged(self) -> None:
        ev = _eval(directives=Directives(max_total_account_invested_usd=3_000.0))
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 200.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
            account_invested_total_usd=2_500.0,
        )
        self.assertTrue(verdicts[0].approved)
        self.assertIsNone(verdicts[0].amended_amount_usd)

    def test_buy_capped_down_when_partially_over(self) -> None:
        ev = _eval(directives=Directives(max_total_account_invested_usd=3_000.0))
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 500.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
            account_invested_total_usd=2_700.0,
        )
        self.assertTrue(verdicts[0].approved)
        # headroom = 3000 - 2700 = 300 → cap to 300
        self.assertEqual(verdicts[0].amended_amount_usd, 300.0)

    def test_buy_rejected_when_cap_exhausted(self) -> None:
        ev = _eval(directives=Directives(max_total_account_invested_usd=2_000.0))
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 200.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
            account_invested_total_usd=2_500.0,
        )
        # Already over the cap thanks to manual positions — bot never
        # tries to fix that by closing them, just refuses to add more.
        self.assertFalse(verdicts[0].approved)
        self.assertIn("max_total_account_invested_usd", verdicts[0].reason)

    def test_close_never_triggers_account_cap(self) -> None:
        # Even when the account is way past the cap, a CLOSE must
        # still go through (so the LLM can rebalance / take profit).
        ev = _eval(directives=Directives(max_total_account_invested_usd=1_000.0))
        state = BotState()
        state.bot_owned_positions.add(42)
        verdicts = ev.evaluate(
            requests=[TradeRequest(99, "AAPL", "CLOSE", 0.0, position_id=42)],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=1,
            account_invested_total_usd=5_000.0,
        )
        self.assertTrue(verdicts[0].approved)

    def test_headroom_below_amend_floor_rejected(self) -> None:
        ev = _eval(
            directives=Directives(max_total_account_invested_usd=3_000.0),
            min_amend_remainder_usd=50.0,
        )
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 200.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
            account_invested_total_usd=2_980.0,
        )
        # headroom = 20 < floor (50) → reject (don't post a dust trade).
        self.assertFalse(verdicts[0].approved)
        self.assertIn("amend floor", verdicts[0].reason)


class DirectivesProviderInvocationTests(unittest.TestCase):
    def test_provider_invoked_per_evaluate_call(self) -> None:
        # The evaluator stores a *callable*, not a snapshot, so a
        # mid-cycle change must be visible on the next evaluate().
        seq = [
            Directives(blocked_symbols=("NVDA",)),
            Directives(),
        ]
        calls = {"count": 0}

        def provider() -> Directives:
            d = seq[calls["count"]]
            calls["count"] += 1
            return d

        ev = RiskEvaluator(
            GuardrailsConfig(
                max_per_trade_usd=500.0, max_parallel_trades=10,
                daily_loss_stop_usd=250.0, per_instrument_cooldown_min=0,
                default_stop_loss_pct=5.0, default_take_profit_pct=8.0,
                max_leverage=1, max_bot_invested_usd=0.0,
                min_amend_remainder_usd=50.0,
            ),
            directives_provider=provider,
        )

        v1 = ev.evaluate(
            requests=[TradeRequest(1, "NVDA", "BUY", 100.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertFalse(v1[0].approved)

        v2 = ev.evaluate(
            requests=[TradeRequest(1, "NVDA", "BUY", 100.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertTrue(v2[0].approved)
        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
