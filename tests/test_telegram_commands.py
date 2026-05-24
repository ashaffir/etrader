"""Tests for Telegram command parsing + dispatch.

We mock out the control API client; the goal is to verify that:
- ``parse_command`` correctly normalizes raw Telegram text.
- Each handler maps to the right control-API method with the right
  arguments.
- Free-text (no leading slash) is forwarded to ``/ask``.
"""

import unittest
from typing import Any

from src.telegram_service.commands import (
    CommandContext,
    dispatch,
    parse_command,
)
from src.telegram_service.control_client import ControlAPIError


class _FakeAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._responses: dict[str, Any] = {}

    def _wrap(self, name: str):
        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self._responses.get(name, {})
        return fn

    def __getattr__(self, name: str):
        return self._wrap(name)

    def queue(self, name: str, response: Any) -> None:
        self._responses[name] = response


class ParseCommandTests(unittest.TestCase):
    def test_parse_plain_command(self) -> None:
        parsed = parse_command("/status")
        self.assertEqual(parsed.name, "status")
        self.assertEqual(parsed.args, "")

    def test_parse_command_with_args(self) -> None:
        parsed = parse_command("/set max_per_trade_usd 250")
        self.assertEqual(parsed.name, "set")
        self.assertEqual(parsed.args, "max_per_trade_usd 250")

    def test_parse_command_with_bot_suffix(self) -> None:
        parsed = parse_command("/status@MyTraderBot")
        self.assertEqual(parsed.name, "status")

    def test_parse_command_normalizes_case(self) -> None:
        parsed = parse_command("/STATUS")
        self.assertEqual(parsed.name, "status")

    def test_non_command_is_treated_as_ask(self) -> None:
        parsed = parse_command("how is the bot doing?")
        self.assertEqual(parsed.name, "ask")
        self.assertEqual(parsed.args, "how is the bot doing?")

    def test_empty_text_returns_empty_command(self) -> None:
        parsed = parse_command("")
        self.assertEqual(parsed.name, "")


def _ctx(api: _FakeAPI, raw: str) -> CommandContext:
    import logging
    log = logging.getLogger("test.telegram")
    log.addHandler(logging.NullHandler())
    return CommandContext(
        api=api,  # type: ignore[arg-type]
        cmd=parse_command(raw),
        sender_username="alice",
        logger=log,
    )


class DispatchTests(unittest.TestCase):
    def test_status_calls_status(self) -> None:
        api = _FakeAPI()
        api.queue("status", {
            "paused": False, "cycle_count": 3, "trading_mode": "paper",
            "env_segment": "demo", "ai_enabled": True,
            "bot_owned_position_count": 0, "tracked_count": 5,
            "base_count": 5, "llm_count": 0, "halted_today": False,
            "halted_day": None, "started_at_unix": 0.0,
            "last_cycle_started_unix": None, "last_cycle_finished_unix": None,
            "last_error": None,
        })
        out = dispatch(_ctx(api, "/status"))
        self.assertEqual([c[0] for c in api.calls], ["status"])
        self.assertIn("[STATUS]", out.text)

    def test_set_requires_two_args(self) -> None:
        api = _FakeAPI()
        out = dispatch(_ctx(api, "/set max_per_trade_usd"))
        self.assertEqual(api.calls, [])
        self.assertIn("Usage: /set", out.text)

    def test_set_passes_key_value(self) -> None:
        api = _FakeAPI()
        api.queue("set_guardrail", {
            "key": "max_per_trade_usd", "previous": 500.0, "current": 250.0,
        })
        dispatch(_ctx(api, "/set max_per_trade_usd 250"))
        self.assertEqual(api.calls[0][0], "set_guardrail")
        self.assertEqual(api.calls[0][1], ("max_per_trade_usd", "250"))

    def test_panic_calls_panic_all(self) -> None:
        api = _FakeAPI()
        api.queue("panic", {
            "scope": "all", "closed_attempted": 0, "closed_ok": 0,
            "results": [], "now_paused": True,
        })
        dispatch(_ctx(api, "/panic"))
        self.assertEqual(api.calls[0][0], "panic")
        self.assertEqual(api.calls[0][2]["scope"], "all")

    def test_panic_bot_only_passes_scope(self) -> None:
        api = _FakeAPI()
        api.queue("panic", {
            "scope": "bot_owned", "closed_attempted": 0, "closed_ok": 0,
            "results": [], "now_paused": True,
        })
        dispatch(_ctx(api, "/panic_bot_only"))
        self.assertEqual(api.calls[0][2]["scope"], "bot_owned")

    def test_stop_aliases_pause(self) -> None:
        api = _FakeAPI()
        api.queue("pause", {"paused": True, "was_already_paused": False})
        dispatch(_ctx(api, "/stop"))
        self.assertEqual([c[0] for c in api.calls], ["pause"])

    def test_start_aliases_resume(self) -> None:
        api = _FakeAPI()
        api.queue("resume", {"paused": False, "was_already_running": False})
        dispatch(_ctx(api, "/start"))
        self.assertEqual([c[0] for c in api.calls], ["resume"])

    def test_history_parses_limit(self) -> None:
        api = _FakeAPI()
        api.queue("history", {"entries": []})
        dispatch(_ctx(api, "/history 5"))
        self.assertEqual(api.calls[0][2], {"limit": 5})

    def test_freeform_text_forwards_to_ask(self) -> None:
        api = _FakeAPI()
        api.queue("ask", {"answer": "all good", "latency_ms": 42})
        out = dispatch(_ctx(api, "what's the bot doing?"))
        self.assertEqual(api.calls[0][0], "ask")
        self.assertEqual(api.calls[0][1], ("what's the bot doing?",))
        self.assertIn("all good", out.text)

    def test_unknown_command_lists_help(self) -> None:
        api = _FakeAPI()
        out = dispatch(_ctx(api, "/widget"))
        self.assertIn("Unknown command", out.text)
        self.assertIn("/status", out.text)

    def test_control_api_error_surfaces_message(self) -> None:
        class _Err:
            def status(self):
                raise ControlAPIError("network down")
        import logging
        log = logging.getLogger("test.telegram.err")
        log.addHandler(logging.NullHandler())
        ctx = CommandContext(
            api=_Err(),  # type: ignore[arg-type]
            cmd=parse_command("/status"),
            sender_username="alice",
            logger=log,
        )
        out = dispatch(ctx)
        self.assertIn("network down", out.text)


if __name__ == "__main__":
    unittest.main()
