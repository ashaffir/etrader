"""Round-trip tests for :class:`src.persistence.StatePersistence`.

The persistence layer must:
- Save and reload bot-owned position IDs verbatim.
- Survive a missing or malformed file without raising.
- Re-project last-action timestamps onto the new process' monotonic
  clock such that elapsed time is preserved (within a few ms).
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.persistence import StatePersistence
from src.state import BotState


class PersistenceRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "bot_state.json"

    def test_round_trip_preserves_owned_positions(self) -> None:
        state = BotState()
        state.add_owned(1001)
        state.add_owned(1002)
        state.session_baseline_equity = 10_000.0
        state.cycle_count = 7
        state.halted_today = True
        state.halted_day = "2026-05-24"
        state.bot_actions_today = 4
        state.baseline_day = "2026-05-24"

        store = StatePersistence(self.path)
        store.save(state, paused=True)

        loaded, meta = store.load()
        self.assertIsNotNone(loaded)
        self.assertIsNotNone(meta)
        assert loaded is not None and meta is not None
        self.assertEqual(loaded.bot_owned_positions, {1001, 1002})
        self.assertEqual(loaded.session_baseline_equity, 10_000.0)
        self.assertEqual(loaded.cycle_count, 7)
        self.assertTrue(loaded.halted_today)
        self.assertEqual(loaded.halted_day, "2026-05-24")
        self.assertEqual(loaded.bot_actions_today, 4)
        self.assertEqual(loaded.baseline_day, "2026-05-24")
        self.assertTrue(meta.paused)

    def test_cooldowns_preserve_elapsed_time(self) -> None:
        state = BotState()
        state.mark_action(42)              # t = ~0
        time.sleep(0.05)                   # +50ms
        marked_at_save = time.monotonic() - state.last_action_per_instrument[42]

        store = StatePersistence(self.path)
        store.save(state, paused=False)

        loaded, _ = store.load()
        assert loaded is not None
        elapsed_after_load = time.monotonic() - loaded.last_action_per_instrument[42]
        # Elapsed time should be approximately preserved (within ~1s tolerance
        # to absorb wall-vs-monotonic clock skew on slow CI).
        self.assertGreaterEqual(elapsed_after_load, marked_at_save)
        self.assertLess(elapsed_after_load - marked_at_save, 1.0)

    def test_missing_file_returns_none(self) -> None:
        store = StatePersistence(self.path)
        loaded, meta = store.load()
        self.assertIsNone(loaded)
        self.assertIsNone(meta)

    def test_malformed_file_returns_none(self) -> None:
        self.path.write_text("not json", encoding="utf-8")
        store = StatePersistence(self.path)
        loaded, meta = store.load()
        self.assertIsNone(loaded)
        self.assertIsNone(meta)

    def test_save_is_atomic(self) -> None:
        store = StatePersistence(self.path)
        store.save(BotState(), paused=False)
        # The .tmp file must NOT remain.
        self.assertFalse(self.path.with_suffix(self.path.suffix + ".tmp").exists())

    def test_handles_missing_keys_gracefully(self) -> None:
        # Older versions might lack newer fields.
        self.path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        loaded, meta = StatePersistence(self.path).load()
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.bot_owned_positions, set())

    def test_autotune_payload_round_trips(self) -> None:
        store = StatePersistence(self.path)
        payload = {
            "cycles_since_last_candidate": 12,
            "cycles_since_last_trade": 240,
            "cycles_since_last_fill": 240,
            "tunings": [
                {"timestamp_unix": 1_700_000_000.0,
                 "reason": "drought",
                 "changes": [{"section": "strategy",
                              "field": "min_signal_strength",
                              "previous": 0.40, "current": 0.25,
                              "rationale": "rolling max 0.30"}]},
            ],
        }
        store.save(BotState(), paused=False, autotune_payload=payload)
        loaded_block = store.load_autotune()
        self.assertEqual(loaded_block, payload)

    def test_legacy_file_has_no_autotune_block(self) -> None:
        store = StatePersistence(self.path)
        store.save(BotState(), paused=False)
        self.assertIsNone(store.load_autotune())


if __name__ == "__main__":
    unittest.main()
