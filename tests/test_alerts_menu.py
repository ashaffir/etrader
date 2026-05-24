"""Tests for the /alerts inline-keyboard rendering + callback parsing."""

from __future__ import annotations

import unittest

from src.telegram_service.alerts_menu import (
    build_alerts_caption,
    build_alerts_keyboard,
    build_closed_caption,
    is_alerts_callback,
    parse_alerts_callback,
)


def _payload(*entries) -> dict:
    return {
        "chat_id": 1,
        "enabled": [e["type"] for e in entries if e["enabled"]],
        "available": list(entries),
    }


class CallbackParserTests(unittest.TestCase):
    def test_is_alerts_callback(self) -> None:
        self.assertTrue(is_alerts_callback("alerts:toggle:trade_opened"))
        self.assertTrue(is_alerts_callback("alerts:close"))
        self.assertFalse(is_alerts_callback("other:do"))
        self.assertFalse(is_alerts_callback(""))

    def test_parse_toggle(self) -> None:
        parsed = parse_alerts_callback("alerts:toggle:trade_opened")
        self.assertEqual(parsed.action, "toggle")
        self.assertEqual(parsed.alert_type, "trade_opened")

    def test_parse_close(self) -> None:
        parsed = parse_alerts_callback("alerts:close")
        self.assertEqual(parsed.action, "close")
        self.assertEqual(parsed.alert_type, "")

    def test_parse_unknown_is_unknown(self) -> None:
        self.assertEqual(parse_alerts_callback("alerts:weird").action, "unknown")
        self.assertEqual(parse_alerts_callback("not_ours:x").action, "unknown")


class KeyboardBuilderTests(unittest.TestCase):
    def test_keyboard_has_one_row_per_type_plus_close(self) -> None:
        payload = _payload(
            {"type": "trade_opened", "label": "Trade opened", "enabled": False},
            {"type": "panic_close", "label": "Panic", "enabled": True},
        )
        kb = build_alerts_keyboard(payload)
        rows = kb["inline_keyboard"]
        self.assertEqual(len(rows), 3)  # 2 types + close
        self.assertEqual(rows[0][0]["callback_data"], "alerts:toggle:trade_opened")
        self.assertEqual(rows[1][0]["callback_data"], "alerts:toggle:panic_close")
        self.assertEqual(rows[2][0]["callback_data"], "alerts:close")

    def test_keyboard_marks_state(self) -> None:
        payload = _payload(
            {"type": "trade_opened", "label": "Trade opened", "enabled": False},
            {"type": "panic_close", "label": "Panic", "enabled": True},
        )
        kb = build_alerts_keyboard(payload)
        on_text = kb["inline_keyboard"][1][0]["text"]
        off_text = kb["inline_keyboard"][0][0]["text"]
        self.assertIn("ON", on_text)
        self.assertIn("OFF", off_text)
        self.assertIn("Trade opened", off_text)
        self.assertIn("Panic", on_text)

    def test_caption_summarizes_state(self) -> None:
        payload = _payload(
            {"type": "trade_opened", "label": "Trade opened", "enabled": True},
            {"type": "panic_close", "label": "Panic", "enabled": False},
        )
        caption = build_alerts_caption(payload)
        self.assertIn("1/2", caption)
        self.assertIn("Tap a row", caption)

    def test_closed_caption_lists_active(self) -> None:
        payload = _payload(
            {"type": "trade_opened", "label": "Trade opened", "enabled": True},
            {"type": "panic_close", "label": "Panic", "enabled": False},
        )
        caption = build_closed_caption(payload)
        self.assertIn("Trade opened", caption)
        self.assertNotIn("Panic", caption)
        self.assertIn("/alerts", caption)

    def test_closed_caption_empty(self) -> None:
        payload = _payload(
            {"type": "trade_opened", "label": "Trade opened", "enabled": False},
        )
        caption = build_closed_caption(payload)
        self.assertIn("all OFF", caption)


if __name__ == "__main__":
    unittest.main()
