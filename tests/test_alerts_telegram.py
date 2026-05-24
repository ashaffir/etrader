"""Tests for /alerts handling end-to-end on the Telegram service side.

Covers:
- ``IncomingCallback`` parsing from a raw Telegram getUpdates payload
- The ``/alerts`` command renders an inline keyboard via ``dispatch``
- The bot loop's callback handler routes a toggle through the control
  API client and edits the keyboard in place
- The drain loop forwards queued alerts as Telegram messages

The Telegram and control APIs are mocked; the focus is on routing /
formatting glue, not the HTTP layer (which has its own tests).
"""

from __future__ import annotations

import logging
import unittest
from typing import Any

from src.telegram_service.bot import TelegramService
from src.telegram_service.commands import (
    CommandContext,
    CommandReply,
    dispatch,
    parse_command,
)
from src.telegram_service.telegram_api import (
    IncomingCallback,
    IncomingMessage,
    _parse_callback,
    _parse_message,
)


class IncomingCallbackParseTests(unittest.TestCase):
    def test_parses_callback_payload(self) -> None:
        cb = _parse_callback(99, {
            "id": "abc",
            "data": "alerts:toggle:panic_close",
            "from": {"id": 7, "username": "alice"},
            "message": {"message_id": 42, "chat": {"id": 100}},
        })
        assert cb is not None
        self.assertEqual(cb.callback_id, "abc")
        self.assertEqual(cb.data, "alerts:toggle:panic_close")
        self.assertEqual(cb.chat_id, 100)
        self.assertEqual(cb.message_id, 42)
        self.assertEqual(cb.user_id, 7)

    def test_callback_without_data_skipped(self) -> None:
        self.assertIsNone(_parse_callback(1, {"id": "x"}))

    def test_message_parser_still_works(self) -> None:
        msg = _parse_message(5, {
            "text": "/status",
            "chat": {"id": 100},
            "from": {"id": 1, "username": "x"},
        })
        assert msg is not None
        self.assertEqual(msg.text, "/status")
        self.assertTrue(msg.is_command)


class _FakeAPI:
    """Minimal control-API client double for /alerts flows."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.subs_payload = {
            "chat_id": 100,
            "enabled": ["panic_close"],
            "available": [
                {"type": "trade_opened", "label": "Trade opened", "enabled": False},
                {"type": "panic_close", "label": "Panic", "enabled": True},
            ],
        }
        self.toggle_payload = {
            "chat_id": 100,
            "type": "trade_opened",
            "enabled": True,
            "all_enabled": ["panic_close", "trade_opened"],
        }
        self.pending_payload: dict[str, Any] = {"alerts": []}

    def alert_subscriptions(self, chat_id: int) -> dict[str, Any]:
        self.calls.append(("alert_subscriptions", (chat_id,), {}))
        return self.subs_payload

    def toggle_alert_subscription(self, chat_id: int, type_str: str) -> dict[str, Any]:
        self.calls.append(("toggle_alert_subscription", (chat_id, type_str), {}))
        return self.toggle_payload

    def alert_pending(self, chat_id: int, *, limit: int = 50) -> dict[str, Any]:
        self.calls.append(("alert_pending", (chat_id,), {"limit": limit}))
        return self.pending_payload


class AlertsCommandTests(unittest.TestCase):
    def _ctx(self, api: _FakeAPI, raw: str, *, chat_id: int = 100) -> CommandContext:
        log = logging.getLogger("test.alerts.cmd")
        log.addHandler(logging.NullHandler())
        return CommandContext(
            api=api,  # type: ignore[arg-type]
            cmd=parse_command(raw),
            sender_username="alice",
            logger=log,
            chat_id=chat_id,
        )

    def test_alerts_command_returns_keyboard(self) -> None:
        api = _FakeAPI()
        reply = dispatch(self._ctx(api, "/alerts"))
        self.assertIsInstance(reply, CommandReply)
        self.assertIn("alert subscriptions", reply.text.lower())
        assert reply.reply_markup is not None
        rows = reply.reply_markup["inline_keyboard"]
        # one row per available type + close
        self.assertEqual(len(rows), 3)
        # Each toggle row has callback_data starting with alerts:toggle:
        toggle_buttons = [rows[0][0], rows[1][0]]
        for btn in toggle_buttons:
            self.assertTrue(btn["callback_data"].startswith("alerts:toggle:"))
        self.assertEqual(rows[-1][0]["callback_data"], "alerts:close")

    def test_alerts_command_without_chat_id_returns_error(self) -> None:
        api = _FakeAPI()
        reply = dispatch(self._ctx(api, "/alerts", chat_id=0))
        self.assertIsNone(reply.reply_markup)
        self.assertIn("chat_id", reply.text)

    def test_subscriptions_alias_works(self) -> None:
        api = _FakeAPI()
        reply = dispatch(self._ctx(api, "/subscriptions"))
        self.assertIsNotNone(reply.reply_markup)


# ---------------------------------------------------------------------------
# Bot-loop integration: callback dispatch + alert drain
# ---------------------------------------------------------------------------

class _FakeTelegram:
    """Captures telegram_api invocations for end-to-end loop tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.acks: list[dict[str, Any]] = []
        self._next_updates: list[Any] = []

    def queue_updates(self, updates: list[Any]) -> None:
        self._next_updates = list(updates)

    def get_updates(self) -> list[Any]:
        out = self._next_updates
        self._next_updates = []
        return out

    def send_message(self, chat_id, text, *, reply_markup=None, parse_mode=None,
                     disable_web_page_preview=True) -> dict[str, Any]:
        record = {
            "chat_id": chat_id, "text": text, "reply_markup": reply_markup,
        }
        self.sent.append(record)
        return {"message_id": 999}

    def edit_reply_markup(self, chat_id, message_id, reply_markup) -> None:
        self.edits.append({
            "chat_id": chat_id, "message_id": message_id,
            "reply_markup": reply_markup,
        })

    def answer_callback(self, callback_id, *, text=None, show_alert=False) -> None:
        self.acks.append({
            "callback_id": callback_id, "text": text, "show_alert": show_alert,
        })


def _service(tg: _FakeTelegram, api: _FakeAPI) -> TelegramService:
    log = logging.getLogger("test.alerts.svc")
    log.addHandler(logging.NullHandler())
    return TelegramService(
        telegram=tg,  # type: ignore[arg-type]
        api=api,      # type: ignore[arg-type]
        allowed_chat_ids=(100,),
        logger=log,
    )


class CallbackDispatchTests(unittest.TestCase):
    def test_toggle_callback_edits_keyboard_and_acks(self) -> None:
        api = _FakeAPI()
        tg = _FakeTelegram()
        svc = _service(tg, api)
        cb = IncomingCallback(
            update_id=1, callback_id="cb1", chat_id=100, message_id=42,
            user_id=7, username="alice", data="alerts:toggle:trade_opened",
        )
        svc._dispatch_update(cb)  # noqa: SLF001 - exercise routing helper
        # Toggle then re-fetch subs then edit_reply_markup.
        names = [c[0] for c in api.calls]
        self.assertEqual(
            names, ["toggle_alert_subscription", "alert_subscriptions"],
        )
        self.assertEqual(len(tg.edits), 1)
        self.assertEqual(tg.edits[0]["chat_id"], 100)
        self.assertEqual(tg.edits[0]["message_id"], 42)
        self.assertIsNotNone(tg.edits[0]["reply_markup"])
        self.assertEqual(len(tg.acks), 1)
        # Ack mentions the new state
        self.assertIn("ON", tg.acks[0]["text"])

    def test_close_callback_strips_keyboard_and_confirms(self) -> None:
        api = _FakeAPI()
        tg = _FakeTelegram()
        svc = _service(tg, api)
        cb = IncomingCallback(
            update_id=1, callback_id="cb1", chat_id=100, message_id=42,
            user_id=7, username="alice", data="alerts:close",
        )
        svc._dispatch_update(cb)  # noqa: SLF001
        # First, fetched subscriptions to render the confirmation message.
        self.assertEqual(api.calls[0][0], "alert_subscriptions")
        # Edit clears keyboard.
        self.assertEqual(len(tg.edits), 1)
        self.assertIsNone(tg.edits[0]["reply_markup"])
        # Confirmation message sent.
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("alerts", tg.sent[0]["text"].lower())
        # Ack happened.
        self.assertEqual(len(tg.acks), 1)

    def test_unknown_callback_just_acks(self) -> None:
        api = _FakeAPI()
        tg = _FakeTelegram()
        svc = _service(tg, api)
        cb = IncomingCallback(
            update_id=1, callback_id="cb1", chat_id=100, message_id=42,
            user_id=7, username="alice", data="other:thing",
        )
        svc._dispatch_update(cb)  # noqa: SLF001
        self.assertEqual(api.calls, [])
        self.assertEqual(tg.edits, [])
        self.assertEqual(len(tg.acks), 1)

    def test_callback_from_disallowed_chat_rejected(self) -> None:
        api = _FakeAPI()
        tg = _FakeTelegram()
        svc = _service(tg, api)
        cb = IncomingCallback(
            update_id=1, callback_id="cb1", chat_id=999, message_id=42,
            user_id=7, username="mallory", data="alerts:toggle:trade_opened",
        )
        svc._dispatch_update(cb)  # noqa: SLF001
        self.assertEqual(api.calls, [])
        self.assertEqual(len(tg.acks), 1)
        self.assertEqual(tg.acks[0]["text"], "Not allowed.")


class AlertDrainTests(unittest.TestCase):
    def test_drain_forwards_each_pending_alert(self) -> None:
        api = _FakeAPI()
        api.pending_payload = {
            "chat_id": 100,
            "alerts": [
                {
                    "type": "panic_close", "title": "PANIC", "body": "all closed",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "type": "trade_opened", "title": "OPENED AAPL",
                    "body": "$250", "timestamp": "2024-01-01T00:01:00Z",
                },
            ],
        }
        tg = _FakeTelegram()
        svc = _service(tg, api)
        svc._drain_alerts_for_all_chats()  # noqa: SLF001
        self.assertEqual(len(tg.sent), 2)
        self.assertIn("PANIC", tg.sent[0]["text"])
        self.assertIn("all closed", tg.sent[0]["text"])
        self.assertIn("OPENED AAPL", tg.sent[1]["text"])
        self.assertEqual(api.calls[0][0], "alert_pending")

    def test_drain_silent_on_api_error(self) -> None:
        from src.telegram_service.control_client import ControlAPIError

        class _BoomAPI(_FakeAPI):
            def alert_pending(self, chat_id, *, limit=50):  # noqa: ARG002
                raise ControlAPIError("control down")

        tg = _FakeTelegram()
        svc = _service(tg, _BoomAPI())
        # Should not raise — just quietly skip this tick.
        svc._drain_alerts_for_all_chats()  # noqa: SLF001
        self.assertEqual(tg.sent, [])


class MessageRoutingTests(unittest.TestCase):
    """The /alerts message path (text → CommandReply with keyboard) must
    flow through the bot's _handle_message and forward the keyboard."""

    def test_alerts_text_command_sends_keyboard(self) -> None:
        api = _FakeAPI()
        tg = _FakeTelegram()
        svc = _service(tg, api)
        msg = IncomingMessage(
            update_id=1, chat_id=100, user_id=7, username="alice",
            text="/alerts", is_command=True,
        )
        svc._handle_message(msg)  # noqa: SLF001
        self.assertEqual(len(tg.sent), 1)
        sent = tg.sent[0]
        self.assertEqual(sent["chat_id"], 100)
        self.assertIsNotNone(sent["reply_markup"])
        self.assertEqual(api.calls[0][0], "alert_subscriptions")


if __name__ == "__main__":
    unittest.main()
