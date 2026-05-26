"""Tests for :mod:`src.ai.decision_context`.

The projection layer is what turns the cycle's tracker + reviewer
state into the dict shape ``build_decision_prompt`` consumes. Tests
exercise:

- per-position enrichment (MFE / MAE / time held / dynamic stops / review)
- per-symbol track-record projection (filter to symbols of interest)
- assembled performance block (None when nothing relevant)
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.ai.decision_context import (
    build_performance_block,
    enrich_owned_position,
    project_bot_owned_positions,
    project_by_symbol_history,
)
from src.etoro.trading import Position
from src.execution.dynamic_stops import DynamicStopsStore
from src.strategy.position_review import PositionReview


def _pos(*, pid: int = 1, iid: int = 1, pnl: float = 0.0) -> Position:
    return Position(
        position_id=pid, instrument_id=iid, is_buy=True,
        open_rate=10.0, amount=100.0, units=10.0, leverage=1,
        mirror_id=0, pnl=pnl, raw={},
    )


class _OpenState:
    """Mock of OpenTradeState — only the fields enrichment reads."""
    def __init__(self, *, pnl_usd: float, pnl_pct: float, mfe: float,
                 mae: float, opened_ago_seconds: float = 120.0) -> None:
        opened = datetime.now(timezone.utc) - timedelta(seconds=opened_ago_seconds)
        self.last_pnl_usd = pnl_usd
        self.last_pnl_pct = pnl_pct
        self.mfe_usd = mfe
        self.mae_usd = mae
        self.opened_at_iso = opened.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.asset_class = "stock"
        self.snapshots = 3


class EnrichOwnedPositionTests(unittest.TestCase):
    def test_bare_position_emits_broker_fields_only(self) -> None:
        out = enrich_owned_position(
            position=_pos(pnl=2.5),
            symbol="AAPL",
            open_state=None,
            dynamic_band=None,
            review=None,
        )
        self.assertEqual(out["instrumentId"], 1)
        self.assertEqual(out["symbol"], "AAPL")
        self.assertEqual(out["positionId"], 1)
        self.assertEqual(out["pnl_usd"], 2.5)
        self.assertNotIn("mfe_usd", out)
        self.assertNotIn("stops", out)
        self.assertNotIn("review", out)

    def test_open_state_adds_perf_block(self) -> None:
        snap = _OpenState(pnl_usd=5.0, pnl_pct=5.0, mfe=8.0, mae=-1.0)
        out = enrich_owned_position(
            position=_pos(),
            symbol="AAPL",
            open_state=snap,
            dynamic_band=None,
            review=None,
        )
        self.assertEqual(out["mfe_usd"], 8.0)
        self.assertEqual(out["mae_usd"], -1.0)
        self.assertEqual(out["pnl_pct"], 5.0)
        self.assertGreater(out["time_held_seconds"], 0)
        self.assertEqual(out["asset_class"], "stock")

    def test_dynamic_band_adds_stops_block(self) -> None:
        store = DynamicStopsStore(default_stop_loss_pct=5.0, default_take_profit_pct=8.0)
        store.set_band(1, stop_loss_pct=2.0, take_profit_pct=6.0,
                       trailing_stop_pct=1.5, rationale="lock")
        band = store.effective_band(1)
        out = enrich_owned_position(
            position=_pos(),
            symbol="AAPL",
            open_state=None,
            dynamic_band=band,
            review=None,
        )
        self.assertEqual(out["stops"]["stop_loss_pct"], 2.0)
        self.assertEqual(out["stops"]["take_profit_pct"], 6.0)
        self.assertEqual(out["stops"]["trailing_stop_pct"], 1.5)
        self.assertEqual(out["stops"]["rationale"], "lock")

    def test_review_block_added(self) -> None:
        review = PositionReview(
            position_id=1, instrument_id=1, symbol="AAPL",
            pnl_usd=-5.0, pnl_pct=-2.5, mfe_usd=2.0, mae_usd=-6.0,
            time_held_seconds=300,
            triggers=["drawdown"],
            notes=["P/L -2.50% breached -2.00%"],
        )
        out = enrich_owned_position(
            position=_pos(), symbol="AAPL",
            open_state=None, dynamic_band=None, review=review,
        )
        self.assertEqual(out["review"]["triggers"], ["drawdown"])
        self.assertEqual(len(out["review"]["notes"]), 1)


class ProjectBotOwnedPositionsTests(unittest.TestCase):
    def test_projection_per_position(self) -> None:
        positions = [_pos(pid=1, iid=10), _pos(pid=2, iid=11)]
        symbol_for = {10: "AAPL", 11: "MSFT"}
        snap_a = _OpenState(pnl_usd=3.0, pnl_pct=3.0, mfe=5.0, mae=0.0)
        store = DynamicStopsStore(default_stop_loss_pct=5.0, default_take_profit_pct=8.0)
        store.set_band(1, stop_loss_pct=2.0)
        out = project_bot_owned_positions(
            positions=positions,
            symbol_for_id=symbol_for,
            open_states={1: snap_a},
            dynamic_stops=store,
        )
        self.assertEqual(len(out), 2)
        # Position 1 has open_state + override:
        self.assertEqual(out[0]["symbol"], "AAPL")
        self.assertIn("mfe_usd", out[0])
        self.assertIn("stops", out[0])
        # Position 2 has neither:
        self.assertEqual(out[1]["symbol"], "MSFT")
        self.assertNotIn("mfe_usd", out[1])
        self.assertNotIn("stops", out[1])


class BySymbolProjectionTests(unittest.TestCase):
    _ROWS = [
        {"symbol": "AAPL", "trades": 3, "wins": 2, "losses": 1,
         "realized_pnl_usd": 12.0, "win_rate": 0.667, "avg_pnl_usd": 4.0,
         "avg_hold_seconds": 1200},
        {"symbol": "MSFT", "trades": 2, "wins": 0, "losses": 2,
         "realized_pnl_usd": -15.0, "win_rate": 0.0, "avg_pnl_usd": -7.5,
         "avg_hold_seconds": 600},
        {"symbol": "TSLA", "trades": 1, "wins": 1, "losses": 0,
         "realized_pnl_usd": 5.0, "win_rate": 1.0, "avg_pnl_usd": 5.0,
         "avg_hold_seconds": 60},
    ]

    def test_filters_to_symbols_of_interest(self) -> None:
        out = project_by_symbol_history(
            by_symbol=self._ROWS,
            symbols_of_interest=["AAPL", "MSFT"],
        )
        self.assertEqual(set(out.keys()), {"AAPL", "MSFT"})
        self.assertNotIn("TSLA", out)

    def test_avg_hold_seconds_converted_to_minutes(self) -> None:
        out = project_by_symbol_history(
            by_symbol=self._ROWS, symbols_of_interest=["MSFT"],
        )
        self.assertEqual(out["MSFT"]["avg_hold_minutes"], 10.0)

    def test_empty_when_no_overlap(self) -> None:
        out = project_by_symbol_history(
            by_symbol=self._ROWS, symbols_of_interest=["NVDA"],
        )
        self.assertEqual(out, {})


class PerformanceBlockTests(unittest.TestCase):
    def test_returns_none_when_nothing_supplied(self) -> None:
        self.assertIsNone(build_performance_block(summary=None))

    def test_assembles_bot_account_period_blocks(self) -> None:
        summary = {
            "bot": {"unrealized_pnl_usd": 3.0, "open_position_count": 1,
                    "trades_total": 5, "realized_pnl_total_usd": -2.5,
                    "realized_pnl_today_usd": -1.0},
            "account": {"unrealized_pnl_usd": 10.0},
            "open": [{"symbol": "AAPL"}],
            "by_period": {
                "today": {"realized_pnl_usd": -1.0, "trades": 1},
            },
        }
        block = build_performance_block(summary=summary)
        self.assertIsNotNone(block)
        self.assertIn("bot", block)
        self.assertIn("account", block)
        self.assertIn("by_period", block)
        self.assertNotIn("open", block)  # stripped

    def test_includes_reviews_and_by_symbol_when_supplied(self) -> None:
        review = PositionReview(
            position_id=1, instrument_id=1, symbol="AAPL",
            pnl_usd=-5.0, pnl_pct=-2.5,
            mfe_usd=2.0, mae_usd=-6.0,
            time_held_seconds=100,
            triggers=["drawdown"],
        )
        block = build_performance_block(
            summary=None,
            reviews=[review],
            by_symbol_projection={"AAPL": {"trades": 1, "wins": 0}},
        )
        self.assertIsNotNone(block)
        self.assertEqual(block["position_reviews"][0]["triggers"], ["drawdown"])
        self.assertEqual(block["by_symbol"]["AAPL"], {"trades": 1, "wins": 0})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
