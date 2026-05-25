"""SQLite-backed config override store.

These tests exercise the store on a temp file so we get the same code
path the production bot uses (no in-memory shortcuts), with one
exception: the resilience tests deliberately hand it a bad path to
prove it falls back to ``:memory:`` instead of crashing.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.config_store import ConfigStore, PERSISTED_SECTIONS, open_store


class StoreLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "config.sqlite"
        self.store = ConfigStore(self.db_path)
        self.addCleanup(self.store.close)

    def test_creates_database_on_first_open(self) -> None:
        self.assertTrue(self.db_path.exists())

    def test_has_any_starts_false(self) -> None:
        self.assertFalse(self.store.has_any())

    def test_get_section_missing_returns_empty(self) -> None:
        self.assertEqual(self.store.get_section("guardrails"), {})

    def test_persisted_sections_constant_is_stable(self) -> None:
        self.assertIn("guardrails", PERSISTED_SECTIONS)
        self.assertIn("strategy", PERSISTED_SECTIONS)


class RoundTripTests(unittest.TestCase):
    """Every JSON-able scalar/list survives a round trip."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "config.sqlite"
        self.store = ConfigStore(self.db_path)
        self.addCleanup(self.store.close)

    def test_float_round_trip(self) -> None:
        self.store.set_field("guardrails", "max_per_trade_usd", 275.5)
        self.assertEqual(
            self.store.get_section("guardrails")["max_per_trade_usd"], 275.5
        )

    def test_int_round_trip(self) -> None:
        self.store.set_field("guardrails", "max_parallel_trades", 4)
        self.assertEqual(
            self.store.get_section("guardrails")["max_parallel_trades"], 4
        )

    def test_bool_round_trip(self) -> None:
        self.store.set_field("universe", "enable_llm_rotation", False)
        self.assertEqual(
            self.store.get_section("universe")["enable_llm_rotation"], False
        )

    def test_list_round_trip(self) -> None:
        self.store.set_field("universe", "base_symbols", ["AAPL", "MSFT"])
        self.assertEqual(
            self.store.get_section("universe")["base_symbols"], ["AAPL", "MSFT"]
        )

    def test_tuple_stored_as_list(self) -> None:
        """Tuples normalise to JSON arrays — the config loader re-tuples them."""
        self.store.set_field("tools", "regime_anchors", ("SPX500", "BTC"))
        decoded = self.store.get_section("tools")["regime_anchors"]
        self.assertEqual(decoded, ["SPX500", "BTC"])

    def test_set_section_writes_all(self) -> None:
        self.store.set_section("guardrails", {
            "max_per_trade_usd": 100.0,
            "max_parallel_trades": 2,
        })
        section = self.store.get_section("guardrails")
        self.assertEqual(section["max_per_trade_usd"], 100.0)
        self.assertEqual(section["max_parallel_trades"], 2)

    def test_upsert_overwrites_existing(self) -> None:
        self.store.set_field("guardrails", "max_per_trade_usd", 500.0)
        self.store.set_field("guardrails", "max_per_trade_usd", 250.0)
        self.assertEqual(
            self.store.get_section("guardrails")["max_per_trade_usd"], 250.0
        )

    def test_delete_field_removes_row(self) -> None:
        self.store.set_field("guardrails", "max_per_trade_usd", 500.0)
        self.store.delete_field("guardrails", "max_per_trade_usd")
        self.assertNotIn("max_per_trade_usd", self.store.get_section("guardrails"))


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "config.sqlite"
        self.store = ConfigStore(self.db_path)
        self.addCleanup(self.store.close)

    def test_snapshot_if_empty_writes_then_skips(self) -> None:
        payload = {
            "guardrails": {"max_per_trade_usd": 500.0, "max_parallel_trades": 10},
            "operations": {"check_interval_seconds": 60},
        }
        first = self.store.snapshot_if_empty(payload)
        self.assertTrue(first)
        self.assertTrue(self.store.has_any())
        self.assertEqual(self.store.get_section("operations")["check_interval_seconds"], 60)

        # Second call must be a no-op: change one field via direct write,
        # then ensure snapshot_if_empty doesn't clobber it.
        self.store.set_field("operations", "check_interval_seconds", 5)
        second = self.store.snapshot_if_empty({
            "operations": {"check_interval_seconds": 999},
        })
        self.assertFalse(second)
        self.assertEqual(self.store.get_section("operations")["check_interval_seconds"], 5)

    def test_snapshot_records_meta(self) -> None:
        self.store.snapshot_if_empty({"guardrails": {"max_per_trade_usd": 1.0}})
        self.assertIsNotNone(self.store.get_meta("first_snapshot_unix"))


class MissingSectionMigrationTests(unittest.TestCase):
    """``add_missing_sections`` backfills late-added sections only."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "config.sqlite"
        self.store = ConfigStore(self.db_path)
        self.addCleanup(self.store.close)

    def test_backfills_only_missing_sections(self) -> None:
        # First-run snapshot only writes the legacy sections.
        self.store.snapshot_if_empty({
            "guardrails": {"max_per_trade_usd": 500.0},
            "strategy": {"min_signal_strength": 0.4},
        })
        # Operator edits one legacy field.
        self.store.set_field("guardrails", "max_per_trade_usd", 123.0)

        # Second-run "migration" introduces fundamentals + keeps legacy.
        added = self.store.add_missing_sections({
            "guardrails": {"max_per_trade_usd": 999.0},  # already present → no-op
            "strategy": {"min_signal_strength": 0.6},     # already present → no-op
            "fundamentals": {"enabled": True, "budget_per_refresh": 8},
        })
        self.assertEqual(added, ["fundamentals"])
        # Existing edit is preserved …
        self.assertEqual(self.store.get_section("guardrails")["max_per_trade_usd"], 123.0)
        # … and the new section is now populated.
        self.assertEqual(
            self.store.get_section("fundamentals"),
            {"enabled": True, "budget_per_refresh": 8},
        )

    def test_idempotent_second_call(self) -> None:
        self.store.snapshot_if_empty({"guardrails": {"x": 1}})
        first = self.store.add_missing_sections({"fundamentals": {"enabled": True}})
        second = self.store.add_missing_sections({"fundamentals": {"enabled": False}})
        self.assertEqual(first, ["fundamentals"])
        self.assertEqual(second, [])
        self.assertTrue(self.store.get_section("fundamentals")["enabled"])

    def test_records_migration_meta(self) -> None:
        self.store.snapshot_if_empty({"guardrails": {"x": 1}})
        self.store.add_missing_sections({"fundamentals": {"enabled": True}})
        self.assertIsNotNone(self.store.get_meta("last_migration_unix"))


class PersistenceAcrossReopenTests(unittest.TestCase):
    """The whole point of the DB: values survive store close/reopen."""

    def test_values_survive_close_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "config.sqlite"

            s1 = open_store(db)
            s1.set_field("guardrails", "default_stop_loss_pct", 7.5)
            s1.set_field("guardrails", "default_take_profit_pct", 12.0)
            s1.close()

            s2 = open_store(db)
            try:
                section = s2.get_section("guardrails")
                self.assertEqual(section["default_stop_loss_pct"], 7.5)
                self.assertEqual(section["default_take_profit_pct"], 12.0)
            finally:
                s2.close()


class ResilienceTests(unittest.TestCase):
    """Bad paths and races must never crash the bot."""

    def test_falls_back_to_memory_on_bad_path(self) -> None:
        store = ConfigStore("/nonexistent_root_dir_etrader/cannot/create/here.sqlite")
        try:
            # The fallback is silent but visible via the path property.
            self.assertEqual(str(store.path), ":memory:")
            store.set_field("guardrails", "max_per_trade_usd", 1.0)
            self.assertEqual(
                store.get_section("guardrails")["max_per_trade_usd"], 1.0
            )
        finally:
            store.close()

    def test_clear_all_wipes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConfigStore(Path(tmp) / "config.sqlite")
            try:
                store.set_field("guardrails", "max_per_trade_usd", 1.0)
                store.set_field("operations", "check_interval_seconds", 60)
                store.clear_all()
                self.assertFalse(store.has_any())
                self.assertEqual(store.get_section("guardrails"), {})
            finally:
                store.close()


class SchemaTests(unittest.TestCase):
    """Sanity-check the on-disk schema."""

    def test_tables_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "config.sqlite"
            store = ConfigStore(db)
            store.close()

            conn = sqlite3.connect(db)
            try:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            finally:
                conn.close()
            names = {r[0] for r in rows}
            self.assertIn("config", names)
            self.assertIn("meta", names)


if __name__ == "__main__":
    unittest.main()
