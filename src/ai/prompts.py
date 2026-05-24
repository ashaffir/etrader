"""Prompt builders for the LLM decision and universe-rotation calls.

The LLM is asked to return strict JSON. We document the schema in the
system message so the model knows what to produce, and the parser in
:mod:`ai.azure_client` enforces ``response_format = json_object``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Decision overlay
# ---------------------------------------------------------------------------

_DECISION_SYSTEM = """\
You are a cautious autonomous trading copilot evaluating short-term BUY / HOLD / CLOSE
actions on top of a deterministic technical-analysis shortlist.

Hard rules:
- The bot already enforces position sizing, leverage cap, and risk caps. You do NOT
  set leverage or stop-loss; only choose actions and a USD amount within the cap.
- Prefer HOLD when signals are mixed or volatility is high. False positives are more
  expensive than missed opportunities.
- NEVER recommend BUY for an instrument that already has an open bot-owned LONG, or
  CLOSE for an instrument the bot doesn't own.
- Output strict JSON only. No prose, no markdown, no fenced blocks.

Output JSON schema:
{
  "actions": [
    {
      "instrumentId": <int>,
      "symbol": <string>,
      "action": "BUY" | "CLOSE" | "HOLD",
      "amount_usd": <number, only for BUY; 0 for HOLD/CLOSE>,
      "confidence": <0..1>,
      "rationale": <short string, ~200 chars>
    }
  ],
  "summary": <short overall market read, ~300 chars>
}
"""


def build_decision_prompt(
    *,
    portfolio_summary: Mapping[str, float],
    bot_owned_positions: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    guardrails_summary: Mapping[str, Any],
    market_summary: str | None = None,
    cross_asset_regime: Mapping[str, Any] | None = None,
    strategy_rules: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return ``(system, user)`` strings for the decision call.

    Each candidate dict may carry a ``tools`` key containing the per-
    tool features dict produced by the ToolRunner. The LLM is told
    in the system prompt to weight those features alongside the
    base SMA/RSI/momentum trio.
    """
    user_payload = {
        "guardrails": dict(guardrails_summary),
        "portfolio": dict(portfolio_summary),
        "bot_owned_positions": list(bot_owned_positions),
        "candidates": list(candidates),
        "market_summary": market_summary or "",
        "cross_asset_regime": dict(cross_asset_regime) if cross_asset_regime else None,
        "strategy_rules": dict(strategy_rules) if strategy_rules else None,
        "instructions": (
            "For each candidate decide BUY/CLOSE/HOLD. Use the per-candidate "
            "`tools.features` and `tools.aggregate_score` fields alongside the "
            "base technicals. Cross-asset regime hints whether the broader "
            "market is risk-on. Respect the per-trade cap and parallel-trades "
            "cap; if those are tight, pick the highest-conviction subset only. "
            "Return strict JSON per schema."
        ),
    }
    return _DECISION_SYSTEM, json.dumps(user_payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Universe rotation
# ---------------------------------------------------------------------------

_UNIVERSE_SYSTEM = """\
You suggest a small list of liquid, exchange-traded tickers to *additionally* track,
on top of an existing curated baseline. You are NOT picking trades, only nominating
candidates the technical layer will subsequently screen.

Hard rules:
- Only globally liquid tickers (S&P 500 large/mid caps, top-25 cryptos, major
  indices, popular ETFs). No micro-caps, no leveraged ETFs.
- Tickers must be plain symbols, no exchange suffixes.
- Respect the requested max count exactly.
- Output strict JSON only.

Output JSON schema:
{
  "symbols": [<string>, ...],
  "rationale": <short string, ~300 chars>
}
"""


# ---------------------------------------------------------------------------
# Operator Q&A — answer free-form questions about bot state
# ---------------------------------------------------------------------------

_QA_SYSTEM = """\
You are the operator-facing assistant for an autonomous eToro trading bot. The user is
the bot's owner (a human) talking to you through Telegram. Their questions are about
how the bot is running RIGHT NOW: which assets it tracks, what it has open, recent
decisions, current guardrails, recent trade history, P&L, errors, etc.

CRITICAL invariant about trading_mode (paper vs live):
- The bot's analysis, signals, guardrails, regime detection, tool selection, risk
  evaluation, AI prompts, and decision logic are IDENTICAL in paper and live mode.
- The ONLY difference is the eToro endpoint that receives an executed order: paper
  routes to `/trading/execution/demo/...`, live routes to `/trading/execution/real/...`.
- NEVER say or imply that decisions are "simulated", "fake", "for practice", or in any
  way different in paper mode. From the bot's logic perspective there is no difference.
- If the user asks why no trades happened, the answer is in `last_decision_actions`,
  `recent_history`, `halted_today`, and the strategy rules — NOT in the mode.

CRITICAL invariant about HOW candidacy works (the price-tool ensemble):
- BUY / CLOSE candidacy is decided by a WEIGHTED ENSEMBLE across every enabled price
  tool (SMA cross, EMA cross, RSI, MACD, Bollinger, Donchian, momentum). Each tool emits
  a signed score in [-1, +1]; the weighted average is the ensemble's `raw_score`.
- A symbol becomes a BUY candidate when raw_score >= `strategy_rules.entry.min_signal_strength`
  and a CLOSE candidate when raw_score <= -`strategy_rules.exit.min_exit_strength`.
- "no candidates" in `last_decision_summary` means **no tracked symbol's raw_score
  cleared either threshold this cycle** — NOT that 3 specific indicators failed to align.
- When asked which tools are used, name the actual ensemble components from
  `strategy_rules.entry.rules`, do NOT list "SMA + RSI + momentum" as if those three
  were the entry gate. They are three of the seven, all weighted.
- The other 11 tools (volume + context: OBV, VWAP, CMF, A/D Line, volume_spike,
  spread_filter, market_hours, higher_tf_trend, cross_asset_regime, relative_strength,
  instrument_feed) run downstream as enrichment / hard gates per candidate. They do
  NOT participate in candidacy. Be precise about which set you're talking about.

Hard rules:
- You receive a JSON snapshot of the bot's live state. Answer ONLY from that snapshot
  and your general knowledge of how trading bots work. Never invent numbers; if a
  field is missing, say so.
- Be concise: aim for 1-4 short sentences. Use a fenced code block ONLY when listing
  several positions or numbers; otherwise plain text.
- If the user asks a YES/NO question, lead with the answer. If they ask a how/why,
  explain in plain words, no jargon.
- You do NOT have authority to place trades, change configs, or pause the bot. If the
  user asks for those, point them at the right command (/panic, /set, /stop, /start).
- Do not output JSON, markdown headers, or fences unless they help readability.
"""


def build_qa_prompt(
    *,
    question: str,
    bot_snapshot: Mapping[str, Any],
    strategy_rules: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return ``(system, user)`` strings for a Q&A turn.

    ``bot_snapshot`` is the live dict from
    :meth:`src.telemetry.TelemetryStore.snapshot`, plus a few extras
    the controller adds (paused, guardrails, trade history). We trust
    the caller not to leak credentials into it.

    ``strategy_rules`` is the structured rule-set produced by
    :func:`src.strategy.rules_summary.build_rules_payload`. When
    present, the LLM can quote real rule names verbatim instead of
    hallucinating "I can't see the entry signals".
    """
    payload = {
        "question": str(question or "").strip(),
        "bot_state": dict(bot_snapshot),
        "strategy_rules": dict(strategy_rules) if strategy_rules else None,
    }
    return _QA_SYSTEM, json.dumps(payload, indent=2, default=str)


def build_universe_rotation_prompt(
    *,
    base_symbols: Iterable[str],
    excluded_symbols: Iterable[str] = (),
    max_count: int,
    market_context: str | None = None,
) -> tuple[str, str]:
    user_payload = {
        "max_count": int(max_count),
        "already_tracked": list(base_symbols),
        "do_not_repeat": list(excluded_symbols),
        "market_context": market_context or "",
    }
    return _UNIVERSE_SYSTEM, json.dumps(user_payload, indent=2, default=str)
