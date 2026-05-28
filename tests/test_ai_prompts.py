"""Tests for :mod:`src.ai.prompts`.

The QA prompt has one invariant we explicitly want to lock in: the LLM
must never imply the bot's logic differs between paper and live mode.
The only difference between the two modes is the eToro endpoint URL
that receives executed orders; everything else (signals, guardrails,
decision logic, regime detection, tool selection) is byte-identical.

If a future contributor edits the system prompt and accidentally
removes that invariant, this test will fail.
"""

from __future__ import annotations

import json
import unittest

from src.ai.prompts import (
    build_decision_prompt,
    build_qa_prompt,
    build_universe_rotation_prompt,
)


class QaPromptModeAgnosticTests(unittest.TestCase):
    def test_qa_system_locks_mode_agnostic_invariant(self) -> None:
        system, _user = build_qa_prompt(
            question="how do i know what the bot is doing?",
            bot_snapshot={"trading_mode": "paper", "env_segment": "demo"},
        )
        s = system.lower()
        self.assertIn("identical in paper and live", s)
        self.assertIn("never say or imply", s)
        self.assertIn("simulated", s)
        # The invariant must reference the actual reasons no trades fired.
        self.assertIn("last_decision_actions", system)
        self.assertIn("halted_today", system)

    def test_qa_system_locks_ensemble_candidacy_invariant(self) -> None:
        """The QA system prompt must explicitly say candidacy is a weighted
        ensemble across every enabled price tool — not 'SMA + RSI + momentum'
        — so the LLM cannot mis-describe the architecture again."""
        system, _user = build_qa_prompt(
            question="why aren't you using more tools?",
            bot_snapshot={},
        )
        s = system.lower()
        self.assertIn("weighted ensemble", s)
        self.assertIn("raw_score", s)
        self.assertIn("min_signal_strength", s)
        self.assertIn("min_exit_strength", s)
        # Mention that volume/context tools run downstream as enrichment/gates,
        # not as candidacy voters — the operator's main confusion.
        self.assertIn("downstream", s)
        for tool in ("sma cross", "ema cross", "macd", "bollinger", "donchian"):
            self.assertIn(tool, s, msg=f"prompt does not name {tool!r}")

    def test_qa_user_payload_includes_snapshot_fields(self) -> None:
        """The bot_state is allowed to be augmented (partitioned positions,
        counts) but the original keys must survive verbatim."""
        snapshot = {"trading_mode": "paper", "cycle_count": 97, "halted_today": False}
        _system, user = build_qa_prompt(
            question="status?",
            bot_snapshot=snapshot,
        )
        payload = json.loads(user)
        for k, v in snapshot.items():
            self.assertEqual(payload["bot_state"][k], v)
        self.assertEqual(payload["question"], "status?")

    def test_qa_payload_partitions_bot_owned_from_manual(self) -> None:
        """Real-world hallucination guard: when the user has 10 positions
        but only 1 is bot-owned, the LLM payload must clearly separate them
        instead of mashing all 10 into a single ``portfolio_positions``."""
        snapshot = {
            "bot_owned_position_ids": [9001],
            "portfolio_positions": [
                {"position_id": 9001, "symbol": "AAPL", "amount": 500.0},
                {"position_id": 9002, "symbol": "MSFT", "amount": 300.0},  # manual
                {"position_id": 9003, "symbol": "TSLA", "amount": 400.0},  # manual
            ],
        }
        _system, user = build_qa_prompt(question="how am I doing?", bot_snapshot=snapshot)
        payload = json.loads(user)
        state = payload["bot_state"]
        self.assertEqual(len(state["bot_owned_positions"]), 1)
        self.assertEqual(state["bot_owned_positions"][0]["symbol"], "AAPL")
        self.assertEqual(len(state["manual_or_mirror_positions"]), 2)
        self.assertEqual(state["counts"]["bot_owned"], 1)
        self.assertEqual(state["counts"]["manual_or_mirror"], 2)
        self.assertEqual(state["counts"]["total_on_account"], 3)
        # The ambiguous original key must be cleared so the LLM can't
        # accidentally pick it up.
        self.assertEqual(state["portfolio_positions"], [])

    def test_qa_system_documents_position_partitioning(self) -> None:
        """The new bot_owned vs manual_or_mirror invariant must be locked in
        the system prompt so a future contributor can't silently remove it."""
        system, _user = build_qa_prompt(question="?", bot_snapshot={})
        s = system.lower()
        self.assertIn("bot_owned_positions", system)
        self.assertIn("manual_or_mirror_positions", system)
        self.assertIn("bot-attributable", s)

    def test_qa_payload_includes_performance_when_supplied(self) -> None:
        """The performance block is the LLM's authoritative source of bot
        P/L numbers; it must round-trip into the payload unchanged."""
        perf = {"bot": {"unrealized_pnl": -12.34, "trades_today": 3}, "account": {"unrealized_pnl": -5.91}}
        _system, user = build_qa_prompt(
            question="how is the bot doing?",
            bot_snapshot={},
            performance=perf,
        )
        payload = json.loads(user)
        self.assertEqual(payload["performance"], perf)

    def test_decision_prompt_does_not_leak_mode(self) -> None:
        """Decision-call prompt must not include a paper/live signal —
        the bot's decisions are mode-agnostic and must reason identically."""
        system, user = build_decision_prompt(
            portfolio_summary={"equity": 10_000.0, "available_cash": 9_000.0},
            bot_owned_positions=[],
            candidates=[],
            guardrails_summary={"max_per_trade_usd": 500.0},
        )
        haystack = (system + user).lower()
        for forbidden in ("paper", "demo", "live", "simulated", "for practice"):
            self.assertNotIn(forbidden, haystack,
                             f"decision prompt unexpectedly contains {forbidden!r}")


class DecisionPromptDynamicManagementTests(unittest.TestCase):
    """The decision prompt must teach the LLM about MODIFY_STOPS,
    partial close, the review.triggers loop, and the by_symbol
    track-record signal. Regression coverage for the dynamic
    position-management surface."""

    def test_system_prompt_documents_modify_stops_action(self) -> None:
        system, _ = build_decision_prompt(
            portfolio_summary={}, bot_owned_positions=[], candidates=[],
            guardrails_summary={},
        )
        self.assertIn("MODIFY_STOPS", system)
        self.assertIn("trailing_stop_pct", system)
        self.assertIn("close_fraction", system)

    def test_system_prompt_documents_review_triggers(self) -> None:
        system, _ = build_decision_prompt(
            portfolio_summary={}, bot_owned_positions=[], candidates=[],
            guardrails_summary={},
        )
        self.assertIn("review.triggers", system)
        self.assertIn("mfe_usd", system)
        self.assertIn("by_symbol", system.lower())

    def test_performance_payload_round_trips(self) -> None:
        perf = {
            "bot": {"unrealized_pnl_usd": 5.0, "trades_total": 10},
            "by_symbol": {"AAPL": {"trades": 3, "win_rate": 0.667}},
            "position_reviews": [
                {"position_id": 1, "triggers": ["drawdown"]},
            ],
        }
        _, user = build_decision_prompt(
            portfolio_summary={}, bot_owned_positions=[], candidates=[],
            guardrails_summary={},
            performance=perf,
        )
        payload = json.loads(user)
        self.assertEqual(payload["performance"], perf)

    def test_omits_performance_when_unset(self) -> None:
        _, user = build_decision_prompt(
            portfolio_summary={}, bot_owned_positions=[], candidates=[],
            guardrails_summary={},
        )
        payload = json.loads(user)
        self.assertIsNone(payload["performance"])


class DecisionPromptExchangeContextTests(unittest.TestCase):
    """The decision system prompt must signpost the per-instrument
    ``exchange`` field so the LLM stops assuming everything is on US
    hours. Without these mentions the multi-market work is invisible
    to the model.
    """

    def test_system_prompt_mentions_exchange_field(self) -> None:
        system, _ = build_decision_prompt(
            portfolio_summary={}, bot_owned_positions=[], candidates=[],
            guardrails_summary={},
        )
        # The DECISION_SYSTEM block now lists exchange labels — both
        # the field's existence and an example of US + foreign labels.
        for label in ("exchange", "NYSE", "LSE", "TSE", "CRYPTO"):
            self.assertIn(label, system)


class UniverseRotationRegionAwareTests(unittest.TestCase):
    """The rotation prompt powers LLM ticker nominations for the
    universe. Two regressions are covered:

    1. The system prompt must allow exchange-suffixed tickers (e.g.
       ``VOD.L``) — the previous version forbade them, capping the
       LLM to US tickers only.
    2. The user payload must include ``currently_open_exchanges`` so
       the LLM can bias nominations toward markets the bot can act
       on right now.
    """

    def test_system_allows_suffixed_tickers(self) -> None:
        system, _ = build_universe_rotation_prompt(
            base_symbols=("AAPL",), excluded_symbols=(), max_count=5,
        )
        # The exact wording of the rule changed; lock the key
        # tokens rather than the prose.
        self.assertIn("exchange suffix", system.lower())
        for example in ("VOD.L", "ASML.AS", "7203.T", "0700.HK"):
            self.assertIn(example, system)

    def test_user_payload_includes_currently_open_exchanges(self) -> None:
        _, user = build_universe_rotation_prompt(
            base_symbols=("AAPL",),
            excluded_symbols=(),
            max_count=3,
            currently_open_exchanges=("NYSE", "LSE", "CRYPTO"),
        )
        payload = json.loads(user)
        self.assertEqual(
            payload["currently_open_exchanges"], ["NYSE", "LSE", "CRYPTO"],
        )

    def test_user_payload_defaults_currently_open_to_empty_list(self) -> None:
        _, user = build_universe_rotation_prompt(
            base_symbols=(), excluded_symbols=(), max_count=1,
        )
        payload = json.loads(user)
        self.assertEqual(payload["currently_open_exchanges"], [])


if __name__ == "__main__":
    unittest.main()
