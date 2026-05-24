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

from src.ai.prompts import build_decision_prompt, build_qa_prompt


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

    def test_qa_user_payload_includes_snapshot_verbatim(self) -> None:
        snapshot = {"trading_mode": "paper", "cycle_count": 97, "halted_today": False}
        _system, user = build_qa_prompt(
            question="status?",
            bot_snapshot=snapshot,
        )
        payload = json.loads(user)
        self.assertEqual(payload["bot_state"], snapshot)
        self.assertEqual(payload["question"], "status?")

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


if __name__ == "__main__":
    unittest.main()
