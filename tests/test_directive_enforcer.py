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
    # 20:58 UTC is 2 min before US equity close at 21:00. Within the
    # default 5-min flatten window.
    IN_WINDOW = datetime(2026, 5, 27, 20, 58, 0, tzinfo=timezone.utc)
    # 13:35 UTC is just after open, well outside the close window.
    NOT_IN_WINDOW = datetime(2026, 5, 27, 13, 35, 0, tzinfo=timezone.utc)
    # Saturday → market closed, no flatten.
    WEEKEND = datetime(2026, 5, 30, 20, 58, 0, tzinfo=timezone.utc)

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

    def test_no_close_on_weekend(self) -> None:
        directives = Directives(no_overnight=True)
        reqs, _notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=[_FakePos(2004, 99)],
            symbol_for_id={99: "AAPL"},
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


class CombinedRulesTests(unittest.TestCase):
    NOW = datetime(2026, 5, 27, 20, 58, 0, tzinfo=timezone.utc)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
