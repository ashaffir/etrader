"""Tests for the /stats Telegram command surface.

Covers:

- Inline-keyboard rendering (button labels, callback_data shape).
- Callback parsing (round-trip).
- Slash-arg shortcut (``/stats today`` etc.) dispatches to the right
  formatter and reaches the right control-API endpoint.
- Formatter sanity for each view.
"""

from __future__ import annotations

import unittest

from src.telegram_service import stats_formatter, stats_menu
from src.telegram_service.commands import (
    CommandContext, _resolve_stats_view_alias, dispatch, parse_command,
    render_stats_view,
)
from src.telegram_service.stats_menu import (
    STATS_VIEWS, VALID_VIEW_IDS, build_stats_keyboard,
    is_stats_callback, parse_stats_callback,
)


# ----------------------------------------------------------------------
# Inline keyboard
# ----------------------------------------------------------------------

class StatsKeyboardTests(unittest.TestCase):
    def test_keyboard_has_a_button_per_view_plus_close(self) -> None:
        kb = build_stats_keyboard()
        all_btns = [b for row in kb["inline_keyboard"] for b in row]
        self.assertEqual(
            len(all_btns), len(STATS_VIEWS) + 1,
            "one button per view plus 'Close'",
        )
        close = next(b for b in all_btns if b["text"] == "Close")
        self.assertEqual(close["callback_data"], "stats:close")

    def test_every_view_callback_round_trips(self) -> None:
        kb = build_stats_keyboard()
        for row in kb["inline_keyboard"]:
            for btn in row:
                data = btn["callback_data"]
                self.assertTrue(is_stats_callback(data))
                parsed = parse_stats_callback(data)
                self.assertIn(parsed.action, ("view", "close"))
                if parsed.action == "view":
                    self.assertIn(parsed.view_id, VALID_VIEW_IDS)

    def test_unknown_callback_rejected(self) -> None:
        self.assertFalse(is_stats_callback("alerts:toggle:TRADE_OPENED"))
        self.assertEqual(parse_stats_callback("stats:view:nope").action, "unknown")
        self.assertEqual(parse_stats_callback("garbage").action, "unknown")


# ----------------------------------------------------------------------
# Slash-arg aliasing
# ----------------------------------------------------------------------

class StatsAliasTests(unittest.TestCase):
    def test_known_aliases_resolve(self) -> None:
        cases = {
            "overview": "overview", "summary": "overview",
            "today": "today", "now": "today",
            "7d": "7d", "week": "7d",
            "30d": "30d", "month": "30d",
            "all": "all", "lifetime": "all",
            "open": "open", "positions": "open",
            "closed": "closed", "trades": "closed",
            "by-symbol": "by_symbol", "symbols": "by_symbol",
            "daily": "daily",
        }
        for alias, expected in cases.items():
            self.assertEqual(
                _resolve_stats_view_alias(alias), expected,
                msg=f"alias {alias!r} did not map to {expected!r}",
            )

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(_resolve_stats_view_alias("blarghh"))


# ----------------------------------------------------------------------
# Dispatch & render
# ----------------------------------------------------------------------

class _StubApi:
    """Minimal ControlAPIClient stand-in for the dispatch tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def stats_summary(self):
        self.calls.append("summary")
        return {
            "enabled": True,
            "bot": {
                "unrealized_pnl_usd": -12.34, "open_position_count": 1,
                "realized_pnl_total_usd": 5.0, "realized_pnl_today_usd": 0.0,
                "trades_total": 3,
            },
            "account": {"unrealized_pnl_usd": -5.91},
            "open": [
                {"symbol": "AMD", "amount_usd": 350.0,
                 "last_pnl_usd": 5.58, "last_pnl_pct": 1.6,
                 "mfe_usd": 6.0, "mae_usd": -1.0},
            ],
            "by_period": {
                "today": {"trades": 0, "wins": 0, "losses": 0,
                          "win_rate_pct": 0.0, "realized_pnl_usd": 0.0,
                          "avg_win_usd": 0.0, "avg_loss_usd": 0.0,
                          "biggest_win_usd": 0.0, "biggest_loss_usd": 0.0,
                          "avg_hold_seconds": 0, "breakeven": 0},
                "7d": {"trades": 1, "wins": 1, "losses": 0,
                       "win_rate_pct": 100.0, "realized_pnl_usd": 5.0,
                       "avg_win_usd": 5.0, "avg_loss_usd": 0.0,
                       "biggest_win_usd": 5.0, "biggest_loss_usd": 0.0,
                       "avg_hold_seconds": 3600, "breakeven": 0},
                "30d": {"trades": 2, "wins": 1, "losses": 1,
                        "win_rate_pct": 50.0, "realized_pnl_usd": 3.0,
                        "avg_win_usd": 5.0, "avg_loss_usd": -2.0,
                        "biggest_win_usd": 5.0, "biggest_loss_usd": -2.0,
                        "avg_hold_seconds": 3600, "breakeven": 0},
                "all": {"trades": 3, "wins": 2, "losses": 1,
                        "win_rate_pct": 66.7, "realized_pnl_usd": 5.0,
                        "avg_win_usd": 4.0, "avg_loss_usd": -3.0,
                        "biggest_win_usd": 5.0, "biggest_loss_usd": -3.0,
                        "avg_hold_seconds": 3600, "breakeven": 0},
            },
        }

    def stats_closed(self, *, limit: int = 50):
        self.calls.append(f"closed:{limit}")
        return {"enabled": True, "rows": [
            {"closed_at_iso": "2026-05-26T13:00:00Z", "symbol": "AMD",
             "realized_pnl_usd": 5.0, "realized_pnl_pct": 1.0, "hold_seconds": 3600},
        ]}

    def stats_by_symbol(self):
        self.calls.append("by_symbol")
        return {"enabled": True, "rows": [
            {"symbol": "AMD", "trades": 1, "win_rate_pct": 100.0, "realized_pnl_usd": 5.0},
        ]}

    def stats_daily(self, *, limit: int = 30):
        self.calls.append(f"daily:{limit}")
        return {"enabled": True, "rows": [
            {"date_iso": "2026-05-26", "equity_close": 104800.0,
             "bot_unrealized_close_usd": -3.18, "bot_trades_today": 1},
        ]}


class _Ctx:
    def __init__(self, api, args: str = "") -> None:
        self.api = api
        self.cmd = parse_command(f"/stats {args}".strip())
        self.sender_username = "tester"
        self.logger = None
        self.chat_id = 1234


class StatsDispatchTests(unittest.TestCase):
    def test_empty_args_returns_keyboard(self) -> None:
        api = _StubApi()
        reply = dispatch(_Ctx(api, args=""))
        self.assertIsNotNone(reply.reply_markup)
        self.assertIn("[STATS]", reply.text)
        # Building the keyboard must not have hit the API.
        self.assertEqual(api.calls, [])

    def test_today_arg_renders_window(self) -> None:
        api = _StubApi()
        reply = dispatch(_Ctx(api, args="today"))
        self.assertIn("[STATS — TODAY]", reply.text)
        self.assertIn("summary", api.calls)

    def test_closed_arg_routes_to_closed_endpoint(self) -> None:
        api = _StubApi()
        reply = dispatch(_Ctx(api, args="closed"))
        self.assertIn("[STATS — CLOSED]", reply.text)
        self.assertTrue(any(c.startswith("closed:") for c in api.calls))

    def test_by_symbol_alias(self) -> None:
        api = _StubApi()
        reply = dispatch(_Ctx(api, args="symbols"))
        self.assertIn("[STATS — BY SYMBOL]", reply.text)
        self.assertIn("by_symbol", api.calls)

    def test_unknown_arg_shows_helpful_message_and_keyboard(self) -> None:
        api = _StubApi()
        reply = dispatch(_Ctx(api, args="blargh"))
        self.assertIn("Unknown", reply.text)
        self.assertIsNotNone(reply.reply_markup)

    def test_render_stats_view_routes_every_view(self) -> None:
        api = _StubApi()
        for vid in VALID_VIEW_IDS:
            text = render_stats_view(api, vid)
            self.assertIsInstance(text, str)
            self.assertIn("[STATS", text)


# ----------------------------------------------------------------------
# Formatter sanity
# ----------------------------------------------------------------------

class StatsFormatterTests(unittest.TestCase):
    def test_disabled_payload(self) -> None:
        text = stats_formatter.format_overview({"enabled": False})
        self.assertIn("not configured", text)

    def test_empty_open_message(self) -> None:
        text = stats_formatter.format_open({"enabled": True, "open": []})
        self.assertIn("no open positions", text)

    def test_window_renders_pnl(self) -> None:
        text = stats_formatter.format_window(
            {"enabled": True, "by_period": {"7d": {
                "trades": 3, "wins": 2, "losses": 1, "breakeven": 0,
                "win_rate_pct": 66.7, "realized_pnl_usd": 12.34,
                "avg_win_usd": 8.0, "avg_loss_usd": -3.0,
                "biggest_win_usd": 10.0, "biggest_loss_usd": -3.0,
                "avg_hold_seconds": 7200,
            }}},
            period="7d",
        )
        self.assertIn("LAST 7 DAYS", text)
        self.assertIn("$12.34", text)


if __name__ == "__main__":
    unittest.main()
