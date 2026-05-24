"""Single source of truth for the strategy rule set.

Both the Telegram ``/signals`` command and the operator-Q&A LLM prompt
read from here, so the bot can never claim "I can't see the entry
rules" again. The summary is generated dynamically from live config
and the registered tool catalog so that flipping a threshold or
adding a tool updates the user-visible explanation automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..config import GuardrailsConfig, StrategyConfig


@dataclass(frozen=True)
class ToolDescription:
    name: str
    family: str           # "price" | "volume" | "context"
    purpose: str
    role: str             # "feature" | "gate"
    asset_classes: tuple[str, ...]


_ENSEMBLE_COMPONENTS = (
    ("sma_cross", "weight_sma_cross",
     "SMA(short)/SMA(long) bull or bear cross within the last 5 candles, "
     "or directional bias from the current spread"),
    ("ema_cross", "weight_ema_cross",
     "EMA(fast)/EMA(slow) bull or bear cross within the last 5 candles"),
    ("rsi", "weight_rsi",
     "RSI: +1 if ≤ oversold, -1 if ≥ overbought, mid-range linearly tilted"),
    ("macd", "weight_macd",
     "MACD histogram zero-cross within the last 3 bars (fallback: sign of histogram)"),
    ("bollinger", "weight_bollinger",
     "Close above upper band → bearish, below lower → bullish, "
     "otherwise position within the band"),
    ("donchian", "weight_donchian",
     "Close above the prior-bar Donchian upper → breakout (+1); "
     "below prior-bar lower → breakdown (-1)"),
    ("momentum", "weight_momentum",
     "Percent change over the lookback, normalized at ±8% saturation"),
)


def _entry_rules(strategy: StrategyConfig) -> list[str]:
    return [
        "Candidate generation is a WEIGHTED ENSEMBLE across price tools, NOT a strict "
        "AND of individual rules.",
        "Each enabled price tool emits a signed score in [-1, +1]; the ensemble takes a "
        "weighted average and the result is the single ``raw_score``.",
        f"BUY candidacy requires: raw_score ≥ {strategy.min_signal_strength:.2f} "
        "(only for instruments the bot does NOT already own).",
        "Per-component contribution detail (name, weight, contribution rule):",
        *[
            f"  - {name} (weight={getattr(strategy, weight_attr):.2f}): {desc}"
            for name, weight_attr, desc in _ENSEMBLE_COMPONENTS
        ],
    ]


def _exit_rules(strategy: StrategyConfig) -> list[str]:
    return [
        f"CLOSE candidacy requires: raw_score ≤ -{strategy.min_exit_strength:.2f} "
        "(only for instruments the bot OWNS).",
        "The same weighted price-tool ensemble drives exits — when the bearish side "
        "of the ensemble outweighs the bullish side, an owned position becomes a "
        "CLOSE candidate.",
        "There is NO separate hard-coded RSI-overbought / SMA-bear-cross / momentum-drop "
        "exit gate; all three feed the ensemble alongside MACD, Bollinger, Donchian, and "
        "EMA cross.",
    ]


def build_rules_payload(
    *,
    strategy: StrategyConfig,
    guardrails: GuardrailsConfig,
    tools: Sequence[ToolDescription] = (),
) -> dict[str, Any]:
    """Return a structured snapshot of the live rule set.

    The structure is stable and machine-friendly so the LLM can quote
    fields exactly when a user asks "what triggers a BUY?".
    """
    return {
        "entry": {
            "trigger": "ALL of",
            "rules": _entry_rules(strategy),
            "min_signal_strength": strategy.min_signal_strength,
        },
        "exit": {
            "trigger": "ANY of (only for bot-owned positions)",
            "rules": _exit_rules(strategy),
        },
        "guardrails": {
            "max_per_trade_usd": guardrails.max_per_trade_usd,
            "max_parallel_trades": guardrails.max_parallel_trades,
            "daily_loss_stop_usd": guardrails.daily_loss_stop_usd,
            "per_instrument_cooldown_min": guardrails.per_instrument_cooldown_min,
            "default_stop_loss_pct": guardrails.default_stop_loss_pct,
            "default_take_profit_pct": guardrails.default_take_profit_pct,
            "max_leverage": guardrails.max_leverage,
        },
        "tools": [
            {
                "name": t.name,
                "family": t.family,
                "purpose": t.purpose,
                "role": t.role,
                "asset_classes": list(t.asset_classes),
            }
            for t in tools
        ],
        "pipeline": [
            "1. Build deterministic candidates from candles (entry/exit rules above).",
            "2. Run the tool selector — picks the relevant subset of tools per instrument "
            "based on asset class, market regime, and recent per-tool performance.",
            "3. Hard gates (e.g. spread_filter, market_hours, trend_filter) can VETO a "
            "candidate before it ever reaches the LLM.",
            "4. Surviving candidates are enriched with feature values from the remaining tools.",
            "5. LLM (if enabled) chooses BUY/HOLD/CLOSE + amount per candidate. "
            "If the LLM is unavailable and veto_on_unavailable=true, the bot HOLDs.",
            "6. RiskEvaluator applies guardrails (per-trade cap, parallel limit, "
            "cooldown, daily-loss kill switch).",
            "7. Approved trades go to the eToro executor. SL/TP are set per guardrails.",
        ],
    }


def render_rules_text(payload: Mapping[str, Any]) -> str:
    """Telegram-friendly multi-line rendering of the rule payload."""
    entry = payload.get("entry") or {}
    exit_ = payload.get("exit") or {}
    g = payload.get("guardrails") or {}
    tools = payload.get("tools") or []

    lines: list[str] = ["[SIGNALS]"]
    lines.append("Entry (BUY) — ALL must hit:")
    for r in entry.get("rules") or []:
        lines.append(f"  - {r}")
    lines.append(f"  - Combined strength >= {entry.get('min_signal_strength', '?')}")
    lines.append("")
    lines.append("Exit (CLOSE) — ANY of (bot-owned only):")
    for r in exit_.get("rules") or []:
        lines.append(f"  - {r}")
    lines.append("")
    lines.append("Guardrails (live):")
    for k, v in g.items():
        lines.append(f"  {k}: {v}")
    if tools:
        lines.append("")
        lines.append(f"Tools registered: {len(tools)}")
        by_family: dict[str, list[str]] = {}
        for t in tools:
            by_family.setdefault(t.get("family", "?"), []).append(t.get("name", "?"))
        for fam in ("price", "volume", "context"):
            if fam in by_family:
                lines.append(f"  {fam:8s}: {', '.join(sorted(by_family[fam]))}")
    lines.append("")
    lines.append("Pipeline:")
    for step in payload.get("pipeline") or []:
        lines.append(f"  {step}")
    return "\n".join(lines)
