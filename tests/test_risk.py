"""Tests for the risk / guardrails layer."""

import unittest

from src.config import GuardrailsConfig
from src.state import BotState
from src.strategy.risk import (
    RiskEvaluator,
    TradeRequest,
    aggregate_summary,
    compute_stop_loss_take_profit,
)


def _eval(**overrides):
    cfg = GuardrailsConfig(
        max_per_trade_usd=overrides.get("max_per_trade_usd", 500.0),
        max_parallel_trades=overrides.get("max_parallel_trades", 10),
        daily_loss_stop_usd=overrides.get("daily_loss_stop_usd", 250.0),
        per_instrument_cooldown_min=overrides.get("per_instrument_cooldown_min", 60),
        default_stop_loss_pct=5.0,
        default_take_profit_pct=8.0,
        max_leverage=1,
        # Disabled by default in the legacy tests so they don't have
        # to reason about the new total-invested budget cap. The
        # BudgetCapTests class exercises it directly.
        max_bot_invested_usd=overrides.get("max_bot_invested_usd", 0.0),
        min_amend_remainder_usd=overrides.get("min_amend_remainder_usd", 50.0),
    )
    return RiskEvaluator(cfg)


class BuyGuardrailTests(unittest.TestCase):
    def test_amount_capped_to_max_per_trade(self) -> None:
        ev = _eval(max_per_trade_usd=200.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 800.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertTrue(verdicts[0].approved)
        self.assertEqual(verdicts[0].amended_amount_usd, 200.0)

    def test_parallel_cap_blocks_excess(self) -> None:
        ev = _eval(max_parallel_trades=2)
        state = BotState()
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "BUY", 100.0),
                TradeRequest(2, "MSFT", "BUY", 100.0),
            ],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=1,  # leaves room for only 1 more
        )
        self.assertTrue(verdicts[0].approved)
        self.assertFalse(verdicts[1].approved)

    def test_cooldown_blocks_recent_instrument(self) -> None:
        state = BotState()
        ev = _eval(per_instrument_cooldown_min=10)
        state.mark_action(1)  # just touched instrument 1
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 100.0)],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertFalse(verdicts[0].approved)
        self.assertIn("cooldown", verdicts[0].reason)

    def test_zero_amount_rejected(self) -> None:
        ev = _eval()
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 0.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertFalse(verdicts[0].approved)


class BudgetCapTests(unittest.TestCase):
    """`max_bot_invested_usd` — caps the bot's total open exposure."""

    def test_disabled_when_cap_is_zero(self) -> None:
        """A cap of 0 must behave identically to the pre-feature bot."""
        ev = _eval(max_bot_invested_usd=0.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 400.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
            bot_invested_total_usd=1_000_000.0,  # absurd; ignored when cap=0
        )
        self.assertTrue(verdicts[0].approved)
        self.assertIsNone(verdicts[0].amended_amount_usd)

    def test_under_cap_passes_unmodified(self) -> None:
        ev = _eval(max_bot_invested_usd=2_000.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 400.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=2,
            bot_invested_total_usd=600.0,  # 600 + 400 = 1000 ≤ 2000
        )
        self.assertTrue(verdicts[0].approved)
        self.assertIsNone(verdicts[0].amended_amount_usd)

    def test_overshoot_is_amended_to_headroom(self) -> None:
        ev = _eval(max_bot_invested_usd=2_000.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 400.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=4,
            bot_invested_total_usd=1_700.0,  # only 300 headroom
        )
        self.assertTrue(verdicts[0].approved)
        self.assertEqual(verdicts[0].amended_amount_usd, 300.0)
        self.assertIn("capped from $400.00", verdicts[0].reason)

    def test_overshoot_below_floor_is_rejected(self) -> None:
        ev = _eval(max_bot_invested_usd=2_000.0, min_amend_remainder_usd=50.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 400.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=4,
            bot_invested_total_usd=1_975.0,  # only $25 headroom < $50 floor
        )
        self.assertFalse(verdicts[0].approved)
        self.assertIn("amend floor", verdicts[0].reason)

    def test_cap_exhausted_rejects(self) -> None:
        ev = _eval(max_bot_invested_usd=1_000.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 100.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=3,
            bot_invested_total_usd=1_000.0,  # at the cap
        )
        self.assertFalse(verdicts[0].approved)
        self.assertIn("budget exhausted", verdicts[0].reason)

    def test_per_trade_cap_applied_first_then_budget(self) -> None:
        """A $800 BUY with max_per_trade=500 and $300 headroom should land at $300."""
        ev = _eval(max_per_trade_usd=500.0, max_bot_invested_usd=2_000.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 800.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=4,
            bot_invested_total_usd=1_700.0,  # 300 headroom
        )
        self.assertTrue(verdicts[0].approved)
        self.assertEqual(verdicts[0].amended_amount_usd, 300.0)

    def test_multiple_buys_share_the_budget_within_a_cycle(self) -> None:
        """Approving BUY #1 must consume headroom for BUY #2 in the same cycle."""
        # Lift max_per_trade so the per-trade cap doesn't kick in first.
        ev = _eval(max_per_trade_usd=1_000.0, max_bot_invested_usd=1_000.0)
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "BUY", 600.0),
                TradeRequest(2, "MSFT", "BUY", 600.0),
            ],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=0,
            bot_invested_total_usd=0.0,
        )
        # BUY #1 fits → approved at 600 untouched.
        self.assertTrue(verdicts[0].approved)
        self.assertIsNone(verdicts[0].amended_amount_usd)
        # BUY #2: only 400 headroom left → amended down (above the $50 floor).
        self.assertTrue(verdicts[1].approved)
        self.assertEqual(verdicts[1].amended_amount_usd, 400.0)


class CloseGuardrailTests(unittest.TestCase):
    def test_close_requires_position_id(self) -> None:
        ev = _eval()
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "CLOSE", 0.0)],
            state=BotState(),
            current_equity=10_000.0,
            bot_owned_position_count=1,
        )
        self.assertFalse(verdicts[0].approved)

    def test_close_must_be_bot_owned(self) -> None:
        ev = _eval()
        state = BotState()
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "CLOSE", 0.0, position_id=42)],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=1,
        )
        self.assertFalse(verdicts[0].approved)
        state.add_owned(42)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "CLOSE", 0.0, position_id=42)],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=1,
        )
        self.assertTrue(verdicts[0].approved)


class PartialCloseGuardrailTests(unittest.TestCase):
    def test_valid_fraction_approved(self) -> None:
        ev = _eval()
        state = BotState()
        state.add_owned(42)
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "CLOSE", 0.0, position_id=42,
                             close_fraction=0.5),
            ],
            state=state, current_equity=10_000.0, bot_owned_position_count=1,
        )
        self.assertTrue(verdicts[0].approved)
        self.assertIn("partial", verdicts[0].reason.lower())

    def test_fraction_outside_band_rejected(self) -> None:
        ev = _eval()
        state = BotState()
        state.add_owned(42)
        for frac in (-0.1, 0.0, 1.5):
            verdicts = ev.evaluate(
                requests=[
                    TradeRequest(1, "AAPL", "CLOSE", 0.0, position_id=42,
                                 close_fraction=frac),
                ],
                state=state, current_equity=10_000.0,
                bot_owned_position_count=1,
            )
            self.assertFalse(
                verdicts[0].approved,
                f"fraction={frac} should be rejected",
            )

    def test_full_close_when_fraction_one(self) -> None:
        ev = _eval()
        state = BotState()
        state.add_owned(42)
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "CLOSE", 0.0, position_id=42,
                             close_fraction=1.0),
            ],
            state=state, current_equity=10_000.0, bot_owned_position_count=1,
        )
        self.assertTrue(verdicts[0].approved)
        self.assertNotIn("partial", verdicts[0].reason.lower())


class ModifyStopsGuardrailTests(unittest.TestCase):
    def test_requires_at_least_one_field(self) -> None:
        ev = _eval()
        state = BotState()
        state.add_owned(42)
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "MODIFY_STOPS", 0.0, position_id=42),
            ],
            state=state, current_equity=10_000.0, bot_owned_position_count=1,
        )
        self.assertFalse(verdicts[0].approved)

    def test_position_must_be_bot_owned(self) -> None:
        ev = _eval()
        state = BotState()
        # No add_owned — position 42 is not bot-owned.
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "MODIFY_STOPS", 0.0, position_id=42,
                             stop_loss_pct=2.0),
            ],
            state=state, current_equity=10_000.0, bot_owned_position_count=0,
        )
        self.assertFalse(verdicts[0].approved)

    def test_out_of_band_value_rejected(self) -> None:
        ev = _eval()
        state = BotState()
        state.add_owned(42)
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "MODIFY_STOPS", 0.0, position_id=42,
                             stop_loss_pct=75.0),  # > 50% — out of band
            ],
            state=state, current_equity=10_000.0, bot_owned_position_count=1,
        )
        self.assertFalse(verdicts[0].approved)

    def test_happy_path_with_all_three_fields(self) -> None:
        ev = _eval()
        state = BotState()
        state.add_owned(42)
        verdicts = ev.evaluate(
            requests=[
                TradeRequest(1, "AAPL", "MODIFY_STOPS", 0.0, position_id=42,
                             stop_loss_pct=3.0, take_profit_pct=7.0,
                             trailing_stop_pct=1.5),
            ],
            state=state, current_equity=10_000.0, bot_owned_position_count=1,
        )
        self.assertTrue(verdicts[0].approved)


class DailyLossKillSwitchTests(unittest.TestCase):
    def test_baseline_set_on_first_eval(self) -> None:
        state = BotState()
        ev = _eval(daily_loss_stop_usd=100.0)
        ev.evaluate(
            requests=[],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertEqual(state.session_baseline_equity, 10_000.0)
        self.assertFalse(state.halted_today)

    def test_drawdown_triggers_halt_when_bot_has_position(self) -> None:
        state = BotState()
        ev = _eval(daily_loss_stop_usd=100.0)
        ev.evaluate(
            requests=[],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=1,  # bot is in the market
        )
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 100.0)],
            state=state,
            current_equity=9_800.0,  # -200 vs baseline → past stop
            bot_owned_position_count=1,
        )
        self.assertTrue(state.halted_today)
        self.assertFalse(verdicts[0].approved)
        self.assertIn("kill switch", verdicts[0].reason)

    def test_drawdown_ignored_when_bot_has_no_skin(self) -> None:
        """Drift in user's manual / mirror positions must not halt the bot."""
        state = BotState()
        ev = _eval(daily_loss_stop_usd=100.0)
        ev.evaluate(
            requests=[],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 100.0)],
            state=state,
            current_equity=9_500.0,  # huge external drawdown
            bot_owned_position_count=0,  # but bot has nothing open
        )
        self.assertFalse(state.halted_today)
        # baseline should rebase to current equity so the bot starts fresh
        self.assertEqual(state.session_baseline_equity, 9_500.0)
        # the BUY is not blocked by the kill switch
        self.assertNotIn("kill switch", verdicts[0].reason)

    def test_stale_persisted_halt_auto_clears_with_no_skin(self) -> None:
        """A halt restored from disk must clear when bot owns 0 and did 0 today."""
        import datetime as dt
        state = BotState()
        state.halted_today = True
        state.halted_day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        state.session_baseline_equity = 10_000.0
        state.baseline_day = state.halted_day
        state.bot_actions_today = 0
        ev = _eval(daily_loss_stop_usd=100.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 100.0)],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertFalse(state.halted_today)
        self.assertNotIn("kill switch", verdicts[0].reason)

    def test_persisted_halt_remains_with_bot_actions_today(self) -> None:
        """If the bot did make actions today, a persisted halt is legitimate."""
        import datetime as dt
        state = BotState()
        state.halted_today = True
        state.halted_day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        state.session_baseline_equity = 10_000.0
        state.baseline_day = state.halted_day
        state.bot_actions_today = 3  # bot did trade today
        ev = _eval(daily_loss_stop_usd=100.0)
        verdicts = ev.evaluate(
            requests=[TradeRequest(1, "AAPL", "BUY", 100.0)],
            state=state,
            current_equity=9_800.0,
            bot_owned_position_count=0,
        )
        self.assertTrue(state.halted_today)
        self.assertFalse(verdicts[0].approved)
        self.assertIn("kill switch", verdicts[0].reason)

    def test_day_rollover_resets_actions_counter(self) -> None:
        state = BotState()
        state.halted_today = True
        state.halted_day = "1999-12-31"
        state.bot_actions_today = 5
        ev = _eval(daily_loss_stop_usd=100.0)
        ev.evaluate(
            requests=[],
            state=state,
            current_equity=10_000.0,
            bot_owned_position_count=0,
        )
        self.assertFalse(state.halted_today)
        self.assertEqual(state.bot_actions_today, 0)
        self.assertEqual(state.session_baseline_equity, 10_000.0)


class StopLossTakeProfitTests(unittest.TestCase):
    def test_long_sl_below_tp_above(self) -> None:
        sl, tp = compute_stop_loss_take_profit(
            entry_price=100.0, is_buy=True, stop_loss_pct=5.0, take_profit_pct=8.0,
        )
        self.assertAlmostEqual(sl, 95.0, places=4)
        self.assertAlmostEqual(tp, 108.0, places=4)

    def test_short_sl_above_tp_below(self) -> None:
        sl, tp = compute_stop_loss_take_profit(
            entry_price=100.0, is_buy=False, stop_loss_pct=5.0, take_profit_pct=8.0,
        )
        self.assertAlmostEqual(sl, 105.0, places=4)
        self.assertAlmostEqual(tp, 92.0, places=4)

    def test_rejects_zero_entry(self) -> None:
        with self.assertRaises(ValueError):
            compute_stop_loss_take_profit(
                entry_price=0.0, is_buy=True, stop_loss_pct=1.0, take_profit_pct=1.0,
            )


class AggregateSummaryTests(unittest.TestCase):
    def test_counts(self) -> None:
        from src.strategy.risk import TradeVerdict  # local import to avoid top-level

        verdicts = [
            TradeVerdict(TradeRequest(1, "A", "BUY", 100.0), True, "ok"),
            TradeVerdict(TradeRequest(2, "B", "BUY", 800.0), True, "ok", amended_amount_usd=500.0),
            TradeVerdict(TradeRequest(3, "C", "BUY", 100.0), False, "denied"),
        ]
        agg = aggregate_summary(verdicts)
        self.assertEqual(agg, {"approved": 2, "denied": 1, "capped": 1})


if __name__ == "__main__":
    unittest.main()
