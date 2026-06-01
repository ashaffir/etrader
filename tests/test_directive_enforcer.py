"""Tests for src/strategy/directive_enforcer.py."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.performance.types import OpenTradeState
from src.strategy.directive_enforcer import (
    DEFAULT_FLATTEN_WINDOW_SECONDS,
    build_directive_close_requests,
    prescreen_candidates,
)
from src.strategy.directives import Directives


# ---------------------------------------------------------------------------
# prescreen_candidates
# ---------------------------------------------------------------------------

@dataclass
class _FakeCandidate:
    symbol: str
    instrument_id: int = 0


class _FakeFund:
    def __init__(self, sector: str) -> None:
        self.sector = sector


class PrescreenTests(unittest.TestCase):
    def test_blocked_symbols_drop_candidate(self) -> None:
        directives = Directives(blocked_symbols=("NVDA",))
        cands = [_FakeCandidate("NVDA"), _FakeCandidate("AAPL")]
        kept, dropped = prescreen_candidates(
            directives=directives, candidates=cands,
        )
        self.assertEqual([c.symbol for c in kept], ["AAPL"])
        self.assertEqual(dropped, [("NVDA", "blocked_symbols (NVDA)")])

    def test_blocked_sector_requires_lookup(self) -> None:
        directives = Directives(blocked_sectors=("Energy",))
        cands = [_FakeCandidate("XOM"), _FakeCandidate("AAPL")]

        def lookup(sym: str):
            return {"XOM": _FakeFund("Energy"), "AAPL": _FakeFund("Technology")}.get(sym)

        kept, dropped = prescreen_candidates(
            directives=directives,
            candidates=cands,
            fundamentals_lookup=lookup,
        )
        self.assertEqual([c.symbol for c in kept], ["AAPL"])
        self.assertEqual(len(dropped), 1)
        self.assertIn("Energy", dropped[0][1])

    def test_no_lookup_keeps_all_when_only_sectors_blocked(self) -> None:
        directives = Directives(blocked_sectors=("Energy",))
        cands = [_FakeCandidate("XOM"), _FakeCandidate("AAPL")]
        kept, dropped = prescreen_candidates(
            directives=directives, candidates=cands, fundamentals_lookup=None,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_empty_directives_passthrough(self) -> None:
        directives = Directives()
        cands = [_FakeCandidate("NVDA"), _FakeCandidate("AAPL")]
        kept, dropped = prescreen_candidates(
            directives=directives, candidates=cands,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])


# ---------------------------------------------------------------------------
# build_directive_close_requests
# ---------------------------------------------------------------------------

@dataclass
class _FakePos:
    position_id: int
    instrument_id: int
    is_buy: bool = True
    amount: float = 100.0


def _open_state(
    *,
    position_id: int,
    instrument_id: int,
    symbol: str,
    asset_class: str,
    opened_minutes_ago: int,
    now: datetime,
) -> OpenTradeState:
    opened_at = (now.timestamp() - opened_minutes_ago * 60)
    opened = datetime.fromtimestamp(opened_at, tz=timezone.utc)
    return OpenTradeState(
        position_id=position_id,
        instrument_id=instrument_id,
        symbol=symbol,
        asset_class=asset_class,
        is_buy=True,
        amount_usd=100.0,
        units=1.0,
        open_rate=100.0,
        opened_at_iso=opened.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


class HoldCeilingTests(unittest.TestCase):
    NOW = datetime(2026, 5, 27, 14, 0, 0, tzinfo=timezone.utc)  # mid-session

    def _setup_state(self, *, opened_minutes_ago: int) -> dict[int, OpenTradeState]:
        return {
            1001: _open_state(
                position_id=1001, instrument_id=42, symbol="AAPL",
                asset_class="stock", opened_minutes_ago=opened_minutes_ago,
                now=self.NOW,
            )
        }

    def test_hold_ceiling_triggers_close(self) -> None:
        directives = Directives(hold_ceiling_minutes=60)
        state = self._setup_state(opened_minutes_ago=120)
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(1001, 42)],
            symbol_for_id={42: "AAPL"},
            instrument_metas={},
            open_states=state,
            now=self.NOW,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].action, "CLOSE")
        self.assertEqual(reqs[0].position_id, 1001)
        self.assertIn("hold_ceiling_minutes", notes[0]["directive"])

    def test_hold_ceiling_below_threshold_no_close(self) -> None:
        directives = Directives(hold_ceiling_minutes=120)
        state = self._setup_state(opened_minutes_ago=30)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(1001, 42)],
            symbol_for_id={42: "AAPL"},
            instrument_metas={},
            open_states=state,
            now=self.NOW,
        )
        self.assertEqual(reqs, [])

    def test_hold_ceiling_zero_disabled(self) -> None:
        directives = Directives(hold_ceiling_minutes=0)
        state = self._setup_state(opened_minutes_ago=600)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(1001, 42)],
            symbol_for_id={42: "AAPL"},
            instrument_metas={},
            open_states=state,
            now=self.NOW,
        )
        self.assertEqual(reqs, [])


class NoOvernightTests(unittest.TestCase):
    # 19:58 UTC = 15:58 EDT = 2 min before US equity close on 2026-05-27
    # (NY summer time). Within the default 5-min flatten window.
    # (Pre-DST-fix the close was hard-coded to 21:00 UTC, which was
    # an hour AFTER the real EDT bell — captured by these tests.)
    IN_WINDOW = datetime(2026, 5, 27, 19, 58, 0, tzinfo=timezone.utc)
    # Equivalent EST scenario: 20:58 UTC on 2026-01-13 (NY winter).
    IN_WINDOW_EST = datetime(2026, 1, 13, 20, 58, 0, tzinfo=timezone.utc)
    # 13:35 UTC = 09:35 EDT — 5 min after the open, well outside close.
    NOT_IN_WINDOW = datetime(2026, 5, 27, 13, 35, 0, tzinfo=timezone.utc)
    # Saturday → market closed, no flatten.
    WEEKEND = datetime(2026, 5, 30, 19, 58, 0, tzinfo=timezone.utc)

    def test_emits_close_for_equity_in_window(self) -> None:
        directives = Directives(no_overnight=True)
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2001, 99)],
            symbol_for_id={99: "AAPL"},
            instrument_metas={},
            now=self.IN_WINDOW,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "no_overnight")

    def test_emits_close_for_equity_in_window_est(self) -> None:
        """Same scenario in NY winter time — close shifts to 21:00 UTC."""
        directives = Directives(no_overnight=True)
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2001, 99)],
            symbol_for_id={99: "AAPL"},
            instrument_metas={},
            now=self.IN_WINDOW_EST,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "no_overnight")

    def test_emits_close_after_real_edt_bell(self) -> None:
        """Regression: 20:30 UTC on an EDT trading day is 30 min AFTER
        the real US bell (16:00 EDT = 20:00 UTC).

        ``no_overnight`` is a hard rule: the bot must not hold equity
        positions outside the regular session. So even after the bell
        has rung, the bot should keep emitting CLOSE requests on every
        cycle until it manages to exit — covering the case where the
        bot was offline during the pre-bell flatten window.
        """
        directives = Directives(no_overnight=True)
        after_bell = datetime(2026, 5, 27, 20, 30, 0, tzinfo=timezone.utc)
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2099, 99)],
            symbol_for_id={99: "AAPL"},
            instrument_metas={},
            now=after_bell,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "no_overnight")
        self.assertIn("closed", notes[0]["reason"])

    def test_skips_crypto_in_window(self) -> None:
        directives = Directives(no_overnight=True)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2002, 999)],
            symbol_for_id={999: "BTC"},
            instrument_metas={},
            now=self.IN_WINDOW,
        )
        # BTC defaults to CRYPTO by symbol heuristic — should be skipped.
        self.assertEqual(reqs, [])

    def test_no_close_outside_window(self) -> None:
        directives = Directives(no_overnight=True)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2003, 99)],
            symbol_for_id={99: "AAPL"},
            instrument_metas={},
            now=self.NOT_IN_WINDOW,
        )
        self.assertEqual(reqs, [])

    def test_emits_close_on_weekend(self) -> None:
        """Weekend = US market closed → the bot must flatten any
        carried-over equity position. This covers the "bot wakes up
        Saturday morning with stale longs" scenario.
        """
        directives = Directives(no_overnight=True)
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2004, 99)],
            symbol_for_id={99: "AAPL"},
            instrument_metas={},
            now=self.WEEKEND,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "no_overnight")
        self.assertIn("closed", notes[0]["reason"])

    def test_emits_close_pre_market(self) -> None:
        """Pre-market (07:00 UTC on a weekday) = market not open yet
        → flatten any leftover overnight position.
        """
        directives = Directives(no_overnight=True)
        pre_market = datetime(2026, 5, 28, 7, 0, 0, tzinfo=timezone.utc)
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2010, 99)],
            symbol_for_id={99: "AAPL"},
            instrument_metas={},
            now=pre_market,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "no_overnight")

    def test_skips_crypto_when_market_closed(self) -> None:
        """Crypto trades 24/7 — closing the equity market doesn't
        require flattening crypto positions.
        """
        directives = Directives(no_overnight=True)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2011, 999)],
            symbol_for_id={999: "BTC"},
            instrument_metas={},
            now=self.WEEKEND,
        )
        self.assertEqual(reqs, [])

    def test_no_close_when_directive_disabled(self) -> None:
        directives = Directives(no_overnight=False)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2005, 99)],
            symbol_for_id={99: "AAPL"},
            instrument_metas={},
            now=self.IN_WINDOW,
        )
        self.assertEqual(reqs, [])


class PerExchangeFlattenTests(unittest.TestCase):
    """Each exchange has its own flatten window — a position on LSE
    shouldn't be force-closed while LSE is mid-session just because
    NY happens to be closed."""

    @staticmethod
    def _meta(price_source: str) -> object:
        m = type("_M", (), {})()
        m.price_source = price_source
        # asset_class_for() also reads stocks_industry_id /
        # instrument_type_id; setting price_source on its own
        # routes to STOCK (via _EQUITY_PRICE_SOURCES).
        m.stocks_industry_id = 1
        m.instrument_type_id = 5
        m.symbol_full = None
        return m

    def test_lse_position_not_flattened_during_lse_hours(self) -> None:
        """11:00 UTC = 12:00 BST (LSE open) — bot must NOT close an LSE position."""
        directives = Directives(no_overnight=True)
        now = datetime(2026, 6, 17, 11, 0, tzinfo=timezone.utc)
        metas = {99: self._meta("lse")}
        reqs, _ = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(8001, 99)],
            symbol_for_id={99: "VOD"},
            instrument_metas=metas,
            now=now,
        )
        self.assertEqual(reqs, [])

    def test_lse_position_flattened_after_lse_close(self) -> None:
        """17:00 UTC = 18:00 BST — LSE closed at 16:30 BST, bot flattens."""
        directives = Directives(no_overnight=True)
        now = datetime(2026, 6, 17, 17, 0, tzinfo=timezone.utc)
        metas = {99: self._meta("lse")}
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(8002, 99)],
            symbol_for_id={99: "VOD"},
            instrument_metas=metas,
            now=now,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "no_overnight")
        self.assertIn("LSE", notes[0]["reason"])

    def test_hkex_position_not_flattened_during_hk_hours(self) -> None:
        """05:00 UTC = 13:00 HKT — HKEX open; NY closed. Position stays."""
        directives = Directives(no_overnight=True)
        now = datetime(2026, 6, 17, 5, 0, tzinfo=timezone.utc)
        metas = {77: self._meta("hkex")}
        reqs, _ = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(8003, 77)],
            symbol_for_id={77: "0700.HK"},
            instrument_metas=metas,
            now=now,
        )
        self.assertEqual(reqs, [])

    def test_tse_position_not_flattened_during_tokyo_hours(self) -> None:
        """03:00 UTC = 12:00 JST — TSE open."""
        directives = Directives(no_overnight=True)
        now = datetime(2026, 6, 17, 3, 0, tzinfo=timezone.utc)
        metas = {55: self._meta("tse")}
        reqs, _ = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(8004, 55)],
            symbol_for_id={55: "7203.T"},
            instrument_metas=metas,
            now=now,
        )
        self.assertEqual(reqs, [])

    def test_nyse_and_lse_positions_treated_independently(self) -> None:
        """11:00 UTC = LSE in-session, NYSE pre-market. Only NYSE position flattens."""
        directives = Directives(no_overnight=True)
        now = datetime(2026, 6, 17, 11, 0, tzinfo=timezone.utc)
        metas = {
            99: self._meta("lse"),
            42: self._meta("nyse"),
        }
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(9001, 99), _FakePos(9002, 42)],
            symbol_for_id={99: "VOD", 42: "AAPL"},
            instrument_metas=metas,
            now=now,
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].position_id, 9002)  # NYSE one only
        self.assertIn("NYSE", notes[0]["reason"])


class CombinedRulesTests(unittest.TestCase):
    # 19:58 UTC on 2026-05-27 = 15:58 EDT, 2 min before US bell.
    NOW = datetime(2026, 5, 27, 19, 58, 0, tzinfo=timezone.utc)

    def test_no_duplicate_close_when_both_rules_apply(self) -> None:
        directives = Directives(
            no_overnight=True, hold_ceiling_minutes=10,
        )
        state = {
            3001: _open_state(
                position_id=3001, instrument_id=11, symbol="AAPL",
                asset_class="stock", opened_minutes_ago=60, now=self.NOW,
            )
        }
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(3001, 11)],
            symbol_for_id={11: "AAPL"},
            instrument_metas={},
            open_states=state,
            now=self.NOW,
        )
        # Both rules match but the position should only close once,
        # tagged with the FIRST matching directive (hold_ceiling).
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "hold_ceiling_minutes")


class PreEarningsCloseTests(unittest.TestCase):
    """``pre_earnings_close_hours`` directive — flatten near earnings."""

    NOW = datetime(2026, 5, 27, 14, 0, 0, tzinfo=timezone.utc)

    def _lookup(self, *, hours_until: float):
        from datetime import timedelta as _td
        from src.strategy.earnings_calendar import EarningsEntry

        when = self.NOW + _td(hours=hours_until)
        entry = EarningsEntry(
            symbol="AAPL",
            earnings_at_utc=when,
            fetched_at_unix=self.NOW.timestamp(),
        )

        def get(_sym: str):
            return entry

        return get

    def test_fires_inside_window(self) -> None:
        directives = Directives(pre_earnings_close_hours=24)
        reqs, notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(4001, 42)],
            symbol_for_id={42: "AAPL"},
            instrument_metas={},
            now=self.NOW,
            earnings_lookup=self._lookup(hours_until=12.0),
        )
        self.assertEqual(len(reqs), 1)
        self.assertEqual(notes[0]["directive"], "pre_earnings_close_hours")
        self.assertIn("earnings", notes[0]["reason"])

    def test_silent_outside_window(self) -> None:
        directives = Directives(pre_earnings_close_hours=24)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(4002, 42)],
            symbol_for_id={42: "AAPL"},
            instrument_metas={},
            now=self.NOW,
            earnings_lookup=self._lookup(hours_until=72.0),
        )
        self.assertEqual(reqs, [])

    def test_disabled_when_threshold_zero(self) -> None:
        directives = Directives(pre_earnings_close_hours=0)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(4003, 42)],
            symbol_for_id={42: "AAPL"},
            instrument_metas={},
            now=self.NOW,
            earnings_lookup=self._lookup(hours_until=2.0),
        )
        self.assertEqual(reqs, [])

    def test_skips_crypto(self) -> None:
        directives = Directives(pre_earnings_close_hours=24)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(4004, 99)],
            symbol_for_id={99: "BTC"},
            instrument_metas={},
            now=self.NOW,
            earnings_lookup=self._lookup(hours_until=2.0),
        )
        self.assertEqual(reqs, [])

    def test_no_lookup_no_close(self) -> None:
        directives = Directives(pre_earnings_close_hours=24)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(4005, 42)],
            symbol_for_id={42: "AAPL"},
            instrument_metas={},
            now=self.NOW,
            earnings_lookup=None,
        )
        self.assertEqual(reqs, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
