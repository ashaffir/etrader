"""Tests for the alerts core: types, subscriptions, hub fan-out."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.alerts import (
    Alert,
    AlertHub,
    AlertSubscriptions,
    AlertType,
    safety_only_default,
)


class AlertTypeTests(unittest.TestCase):
    def test_all_types_listed(self) -> None:
        names = {t.value for t in AlertType.all_types()}
        self.assertIn("trade_opened", names)
        self.assertIn("trade_closed", names)
        self.assertIn("panic_close", names)
        self.assertIn("daily_loss_halt", names)
        self.assertIn("cycle_error", names)
        self.assertIn("ai_unavailable", names)
        self.assertIn("universe_changed", names)
        self.assertIn("bot_paused_resumed", names)
        self.assertIn("trade_failed", names)
        self.assertIn("universe_rejected", names)
        self.assertIn("strategy_autotuned", names)
        self.assertIn("order_stuck_cant_cancel", names)
        self.assertEqual(len(names), 12)
        self.assertNotIn("cycle_heartbeat", names)

    def test_safety_only_default_subset(self) -> None:
        defaults = safety_only_default()
        self.assertEqual(defaults, {
            AlertType.PANIC_CLOSE,
            AlertType.DAILY_LOSS_HALT,
            AlertType.CYCLE_ERROR,
            AlertType.TRADE_FAILED,
            # The autotuner edits real bot behaviour; we want operators
            # to always see when it fires, so it's seeded ON by default.
            AlertType.STRATEGY_AUTOTUNED,
            # Stuck orders that can't be auto-cancelled require manual
            # intervention.
            AlertType.ORDER_STUCK_CANT_CANCEL,
        })

    def test_from_value_unknown_returns_none(self) -> None:
        self.assertIsNone(AlertType.from_value("nope"))
        self.assertEqual(AlertType.from_value("trade_opened"), AlertType.TRADE_OPENED)

    def test_universe_rejected_is_opt_in(self) -> None:
        # ``universe_rejected`` is intentionally NOT in the safety-only
        # default because it can be chatty when the activity filter is
        # tuned tight. Operators opt in via /alerts.
        self.assertNotIn(AlertType.UNIVERSE_REJECTED, safety_only_default())


class AlertSubscriptionsTests(unittest.TestCase):
    def test_default_seeded_on_first_read(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "subs.json"
            subs = AlertSubscriptions(path, default_set=safety_only_default())
            enabled = subs.enabled_for(123)
            self.assertEqual(enabled, safety_only_default())
            self.assertTrue(path.exists())

    def test_persisted_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "subs.json"
            subs = AlertSubscriptions(path, default_set=safety_only_default())
            subs.set_enabled(42, AlertType.TRADE_OPENED, True)
            subs.set_enabled(42, AlertType.PANIC_CLOSE, False)
            # Reload
            subs2 = AlertSubscriptions(path, default_set=safety_only_default())
            enabled = subs2.enabled_for(42)
            self.assertIn(AlertType.TRADE_OPENED, enabled)
            self.assertNotIn(AlertType.PANIC_CLOSE, enabled)

    def test_toggle_flips_state_and_returns_new_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            subs = AlertSubscriptions(
                Path(td) / "s.json", default_set={AlertType.TRADE_FAILED},
            )
            new_state, full = subs.toggle(7, AlertType.TRADE_OPENED)
            self.assertTrue(new_state)
            self.assertIn(AlertType.TRADE_OPENED, full)
            new_state, full = subs.toggle(7, AlertType.TRADE_OPENED)
            self.assertFalse(new_state)
            self.assertNotIn(AlertType.TRADE_OPENED, full)

    def test_load_skips_unknown_types_and_chat_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "subs.json"
            path.write_text(
                json.dumps({
                    "9": ["trade_opened", "not_a_real_type"],
                    "abc": ["trade_failed"],  # bad chat id
                }),
                encoding="utf-8",
            )
            subs = AlertSubscriptions(path, default_set=safety_only_default())
            self.assertEqual(subs.enabled_for(9), {AlertType.TRADE_OPENED})

    def test_reset_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            subs = AlertSubscriptions(
                Path(td) / "s.json", default_set=safety_only_default(),
            )
            subs.set_enabled(11, AlertType.TRADE_OPENED, True)
            subs.set_enabled(11, AlertType.PANIC_CLOSE, False)
            new = subs.reset_to_default(11)
            self.assertEqual(new, safety_only_default())


class AlertHubTests(unittest.TestCase):
    def _make(self, chat_ids=(101,), default_set=None):
        td = tempfile.mkdtemp(prefix="alerts_")
        subs = AlertSubscriptions(
            Path(td) / "s.json",
            default_set=default_set or safety_only_default(),
        )
        return AlertHub(allowed_chat_ids=chat_ids, subscriptions=subs)

    def test_emit_only_to_subscribed_chats(self) -> None:
        hub = self._make(chat_ids=(101, 202))
        # 101 keeps default (safety only); 202 unsubscribes from PANIC_CLOSE
        hub.subscriptions.set_enabled(202, AlertType.PANIC_CLOSE, False)

        hub.emit(AlertType.PANIC_CLOSE, title="X", body="y")
        a101 = hub.drain(101)
        a202 = hub.drain(202)

        self.assertEqual(len(a101), 1)
        self.assertEqual(a101[0].type, AlertType.PANIC_CLOSE)
        self.assertEqual(a202, [])

    def test_emit_skipped_when_no_chats_configured(self) -> None:
        hub = self._make(chat_ids=())
        # Should be a no-op, no crash.
        hub.emit(AlertType.CYCLE_ERROR, title="boom")
        # No queues, so depth lookup returns 0.
        self.assertEqual(hub.queue_depth(123), 0)

    def test_drain_respects_limit_and_order(self) -> None:
        hub = self._make()
        for i in range(5):
            hub.emit(AlertType.PANIC_CLOSE, title=f"p{i}")
        first_two = hub.drain(101, limit=2)
        self.assertEqual([a.title for a in first_two], ["p0", "p1"])
        rest = hub.drain(101, limit=10)
        self.assertEqual([a.title for a in rest], ["p2", "p3", "p4"])

    def test_alert_to_dict_round_trip_safe(self) -> None:
        a = Alert(
            type=AlertType.TRADE_OPENED,
            timestamp="2024-01-01T00:00:00Z",
            title="t",
            body="b",
        )
        d = a.to_dict()
        self.assertEqual(d["type"], "trade_opened")
        self.assertEqual(d["title"], "t")
        self.assertEqual(d["body"], "b")
        self.assertEqual(d["timestamp"], "2024-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
