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
