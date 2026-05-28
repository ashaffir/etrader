"""Prompt builders for the LLM decision and universe-rotation calls.

The LLM is asked to return strict JSON. We document the schema in the
system message so the model knows what to produce, and the parser in
:mod:`ai.azure_client` enforces ``response_format = json_object``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .prompt_texts import DECISION_SYSTEM as _DECISION_SYSTEM


def build_decision_prompt(
    *,
    portfolio_summary: Mapping[str, float],
    bot_owned_positions: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    guardrails_summary: Mapping[str, Any],
    market_summary: str | None = None,
    cross_asset_regime: Mapping[str, Any] | None = None,
    strategy_rules: Mapping[str, Any] | None = None,
    autotune_evidence: Mapping[str, Any] | None = None,
    performance: Mapping[str, Any] | None = None,
    directives: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return ``(system, user)`` strings for the decision call.

    ``performance`` (when provided) is the projection produced by
    :func:`src.ai.decision_context.build_performance_block` — the
    aggregate scoreboard + per-symbol track record + the active
    position-review annotations. The LLM uses it to decide whether
    open positions should be closed, partially closed, or have their
    SL/TP modified.

    ``autotune_evidence`` is the digest produced by
    :class:`src.strategy.autotune.AutotuneState.build_evidence`. The
    LLM uses it to decide whether to attach an OPTIONAL ``tuning``
    block to its response — see the system prompt for the schema.

    ``directives`` is the operator-set persistent ruleset returned by
    :meth:`BotController.snapshot_directives`. Hard rules (blocked
    symbols, sectors, total-cap, etc.) are also enforced by the risk
    layer; the LLM still sees them so it can avoid wasting BUY slots
    on something the risk layer is going to reject, and so it can
    weigh free-text ``notes`` for soft preferences.
    """
    user_payload = {
        "guardrails": dict(guardrails_summary),
        "directives": dict(directives) if directives else None,
        "portfolio": dict(portfolio_summary),
        "bot_owned_positions": list(bot_owned_positions),
        "candidates": list(candidates),
        "performance": dict(performance) if performance else None,
        "market_summary": market_summary or "",
        "cross_asset_regime": dict(cross_asset_regime) if cross_asset_regime else None,
        "strategy_rules": dict(strategy_rules) if strategy_rules else None,
        "autotune_evidence": dict(autotune_evidence) if autotune_evidence else None,
        "instructions": (
            "Decide per-instrument actions covering BOTH new candidates AND "
            "every existing bot-owned position. Honour every active operator "
            "directive in `directives` — they are persistent rules the user "
            "set explicitly; never propose a BUY that violates one and "
            "respect the free-text `notes` as soft preferences. For each "
            "open position with a non-empty `review.triggers`, emit CLOSE, "
            "MODIFY_STOPS, or HOLD with explicit rationale. For new "
            "candidates use BUY (size with `amount_usd`) or HOLD. Consult "
            "`performance.by_symbol` before BUYing a symbol with a poor "
            "track record — either size down or skip. Inspect "
            "`autotune_evidence`: if the rolling raw_score distribution "
            "and drought counters indicate the entry gate is mis-calibrated, "
            "include a `tuning` block per the schema. Most cycles should "
            "omit it. Return strict JSON per schema."
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

The user message includes `currently_open_exchanges` — bias your nominations toward
markets that are open RIGHT NOW. Suggesting US names while only Tokyo and Hong Kong
are open means the bot can't actually act on them this cycle.

Hard rules:
- Globally liquid tickers only (S&P 500 large/mid caps, FTSE 100, DAX 40, CAC 40,
  Euro Stoxx 50, Nikkei 225, Hang Seng, ASX 200 leaders, top-25 cryptos, major
  ETFs). No micro-caps, no leveraged ETFs.
- Tickers must include the exchange suffix where eToro requires it (e.g. ``VOD.L``,
  ``ASML.AS``, ``SAP.DE``, ``7203.T``, ``0700.HK``, ``BHP.AX``). Plain symbols are
  fine for US-listed names where no suffix is needed (``AAPL``, ``NVDA``).
- Aim for regional diversity. Don't return 10 US names when foreign markets are open.
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

CRITICAL invariant about WHICH positions are the bot's:
- The eToro account may hold positions the user opened by hand or copied from a
  popular investor (mirror positions). The bot did NOT open those — it neither
  monitors their entries nor will it close them.
- The JSON payload now has TWO disjoint position arrays:
    * `bot_state.bot_owned_positions` — opened by THIS bot. Their P/L counts
      against the bot's track record.
    * `bot_state.manual_or_mirror_positions` — opened by the user manually or
      mirrored from a copy-trader. Their P/L is the USER's, NOT the bot's.
- `bot_state.counts.bot_owned` is the authoritative count for "how many positions
  does the bot have open". NEVER use the total or `manual_or_mirror` count when
  asked about the bot.
- When the user asks "how is the bot doing?", "what's my P/L?", "how many positions
  do we have?", scope the answer to `bot_owned_positions` and the bot-attributable
  numbers in `performance.bot`. If you want to also mention overall account state,
  label it explicitly: "Bot: X. Whole account incl. your manual trades: Y."

CRITICAL invariant about EXCHANGES (multi-market trading):
- The bot trades across many exchanges, not just US markets. Each position and
  each candidate in the payload now carries an `exchange` field (e.g. ``"NYSE"``,
  ``"NASDAQ"``, ``"LSE"``, ``"XETRA"``, ``"HKEX"``, ``"TSE"``, ``"ASX"``,
  ``"CRYPTO"``, ``"FX"``).
- When the user asks "which exchanges?", "where are my positions?", or "what
  markets is the bot trading?", group BY THIS FIELD — do NOT hand-classify
  symbols by guessing from the ticker. If `exchange` is ``null`` for a position
  it means the bot doesn't have metadata for that instrument this cycle
  (typically a manually opened position on a non-tracked name); say so
  explicitly rather than guessing.
- The market-open check is also per-exchange now. "Is the market open?" for the
  bot's portfolio is answered per-position: an LSE name is open during London
  hours even if NY is closed. Cross-reference with the time-of-day if asked.

CRITICAL invariant about OPERATOR DIRECTIVES:
- The `directives` block lists PERSISTENT rules the user attached via Telegram. Honour
  them in your answers and decisions: when asked "why didn't the bot buy NVDA?" check
  `directives.values.blocked_symbols` first; when asked "will positions stay open
  overnight?" check `directives.values.no_overnight`.
- The free-text `directives.values.notes` field is the operator talking directly to
  you. Quote it (or paraphrase) when answering questions like "any preferences from
  the operator?" — it is the source of truth for soft directives.
- Directives are persistent across bot restarts. Mention them when explaining recent
  refusals or scheduled closes.

CRITICAL invariant about P/L numbers:
- The `performance` block (when present) contains pre-computed bot-attributable
  P/L over multiple windows (today, 7d, 30d, all-time). Use these directly.
  Do NOT add up trade-history entries by hand — the realized P/L on most older
  rows is unknown and you'll under- or over-report.
- `performance.account.unrealized_pnl` is the WHOLE account's unrealized P/L
  (including the user's manual positions). `performance.bot.unrealized_pnl` is
  ONLY what the bot's open positions are doing. When the user asks "am I losing
  money", clarify which scope you mean.

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
    performance: Mapping[str, Any] | None = None,
    directives: Mapping[str, Any] | None = None,
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

    ``performance`` is the structured performance payload produced by
    :meth:`src.performance.PerformanceTracker.summary`. Without it the
    LLM has to read raw trade history to compute P/L, which is where
    the hallucinated "9 bot-owned positions losing $450" answer came
    from. With it, the LLM has authoritative bot-attributable numbers
    pre-computed.
    """
    bot_state = dict(bot_snapshot)
    bot_state = _partition_positions(bot_state)
    payload = {
        "question": str(question or "").strip(),
        "bot_state": bot_state,
        "strategy_rules": dict(strategy_rules) if strategy_rules else None,
        "performance": dict(performance) if performance else None,
        "directives": dict(directives) if directives else None,
    }
    return _QA_SYSTEM, json.dumps(payload, indent=2, default=str)


def _partition_positions(bot_state: dict[str, Any]) -> dict[str, Any]:
    """Split ``portfolio_positions`` into bot-owned vs manual/mirror.

    The original payload handed the LLM ``portfolio_positions`` (every
    open position on the eToro account, including ones the user opened
    by hand or mirrored from a copy-trader) alongside
    ``bot_owned_position_ids`` (the subset the bot is responsible for).
    The LLM repeatedly conflated the two, claiming things like "you
    have 9 bot-owned positions" when the bot owned 0. Splitting them
    in the payload — and labelling each half explicitly — removes the
    ambiguity. The LLM no longer needs to cross-reference IDs.
    """
    bot_owned_ids = {int(x) for x in (bot_state.get("bot_owned_position_ids") or [])}
    all_positions = list(bot_state.get("portfolio_positions") or [])
    bot_positions: list[Any] = []
    other_positions: list[Any] = []
    for p in all_positions:
        if not isinstance(p, dict):
            continue
        pid_raw = p.get("position_id") or p.get("positionID") or p.get("positionId")
        try:
            pid = int(pid_raw) if pid_raw is not None else None
        except (TypeError, ValueError):
            pid = None
        if pid is not None and pid in bot_owned_ids:
            bot_positions.append(p)
        else:
            other_positions.append(p)
    # Replace the ambiguous ``portfolio_positions`` with two explicit
    # buckets. We keep the original key empty so any older prompts that
    # rely on it can't silently pick up the wrong set.
    bot_state["bot_owned_positions"] = bot_positions
    bot_state["manual_or_mirror_positions"] = other_positions
    bot_state["portfolio_positions"] = []  # intentionally cleared
    bot_state["counts"] = {
        "bot_owned": len(bot_positions),
        "manual_or_mirror": len(other_positions),
        "total_on_account": len(bot_positions) + len(other_positions),
    }
    return bot_state


def build_universe_rotation_prompt(
    *,
    base_symbols: Iterable[str],
    excluded_symbols: Iterable[str] = (),
    max_count: int,
    market_context: str | None = None,
    currently_open_exchanges: Iterable[str] = (),
) -> tuple[str, str]:
    """Return ``(system, user)`` for the universe-rotation LLM call.

    ``currently_open_exchanges`` is a list of exchange labels (e.g.
    ``["NYSE", "LSE"]``) the cycle considers in-session at the time of
    the call. The LLM uses it to bias its nominations toward markets
    the bot can actually trade right now, instead of always suggesting
    US names regardless of the wall-clock hour.
    """
    user_payload = {
        "max_count": int(max_count),
        "already_tracked": list(base_symbols),
        "do_not_repeat": list(excluded_symbols),
        "market_context": market_context or "",
        "currently_open_exchanges": list(currently_open_exchanges),
    }
    return _UNIVERSE_SYSTEM, json.dumps(user_payload, indent=2, default=str)
