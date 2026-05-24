"""Smoke tests for the in-memory telemetry store.

We don't drive concurrency here (the GIL + simple lock makes it
trivially safe); we just verify the snapshot reflects the latest writes
and that copies are deep enough to insulate readers from later
mutations of the original data.
"""

import unittest

from src.telemetry import TelemetryStore


class TelemetryStoreTests(unittest.TestCase):
    def test_round_trip_status(self) -> None:
        store = TelemetryStore()
        store.mark_cycle_started(5)
        store.update_universe(
            instrument_ids=[1, 2, 3],
            symbols=["A", "B", "C"],
            base_count=2,
            llm_count=1,
        )
        store.update_decision(
            summary="market quiet",
            llm_used=True,
            actions=[{"action": "BUY", "symbol": "A", "amount_usd": 100.0}],
        )
        store.mark_cycle_finished()

        snap = store.snapshot()
        self.assertEqual(snap["cycle_count"], 5)
        self.assertEqual(snap["tracked_symbols"], ["A", "B", "C"])
        self.assertEqual(snap["base_count"], 2)
        self.assertTrue(snap["last_decision_llm_used"])
        self.assertEqual(snap["last_decision_actions"][0]["symbol"], "A")
        self.assertIsNotNone(snap["last_cycle_finished_unix"])

    def test_snapshot_isolates_callers(self) -> None:
        store = TelemetryStore()
        store.update_portfolio(
            summary={"equity": 1000.0},
            positions=[{"symbol": "A", "amount": 50.0}],
            bot_owned_position_ids=[7],
        )
        snap = store.snapshot()
        snap["portfolio_positions"][0]["amount"] = 9999  # mutate the copy
        snap["bot_owned_position_ids"].append(99)

        snap2 = store.snapshot()
        self.assertEqual(snap2["portfolio_positions"][0]["amount"], 50.0)
        self.assertEqual(snap2["bot_owned_position_ids"], [7])

    def test_mark_cycle_error_records_message(self) -> None:
        store = TelemetryStore()
        store.mark_cycle_started(1)
        store.mark_cycle_error("boom")
        snap = store.snapshot()
        self.assertEqual(snap["last_error"], "boom")
        self.assertIsNotNone(snap["last_cycle_finished_unix"])


if __name__ == "__main__":
    unittest.main()
