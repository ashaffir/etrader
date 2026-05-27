"""Renderer for the /tokens Telegram surface.

Shows three windows (today / 7d / all-time), a per-call-type
breakdown, and the current deployment + per-1M-token rates the
controller used to estimate dollar cost. Missing rates surface as
"—" so the operator knows the table is incomplete (instead of
seeing a misleading $0).
"""

from __future__ import annotations

from typing import Any, Mapping


def format_tokens(payload: Mapping[str, Any]) -> str:
    if not payload.get("enabled", True):
        return (
            "[TOKENS]\n"
            "LLM usage tracking is disabled (AI not configured).\n"
            "Configure AZURE_OPENAI_* env vars to enable."
        )

    deployment = str(payload.get("deployment") or "(unknown)")
    rates = payload.get("rates")
    lines = ["[TOKENS]"]
    lines.append(f"deployment: {deployment}")
    if rates:
        lines.append(
            f"rates ($/1M tokens): input {_fmt_money(rates.get('input_per_m'))}, "
            f"cached {_fmt_money(rates.get('cached_per_m'))}, "
            f"output {_fmt_money(rates.get('output_per_m'))}"
        )
    else:
        lines.append("rates ($/1M tokens): — (no entry for this deployment)")
    lines.append("")

    for window_key, label in (
        ("today", "TODAY"),
        ("last_7d", "7 DAYS"),
        ("last_30d", "30 DAYS"),
        ("all_time", "ALL-TIME"),
    ):
        window = payload.get(window_key)
        if not isinstance(window, dict):
            continue
        lines.append(f"— {label} —")
        lines.append(_fmt_window(window))
        lines.append("")

    by_call = payload.get("by_call_type") or {}
    if by_call:
        lines.append("— BY CALL TYPE (today) —")
        for call_type, stats in sorted(by_call.items()):
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"  {call_type:14s} "
                f"calls={int(stats.get('calls') or 0):4d}  "
                f"in={int(stats.get('prompt_tokens') or 0):8,}  "
                f"out={int(stats.get('completion_tokens') or 0):7,}  "
                f"cost={_fmt_money(stats.get('cost_usd'))}"
            )
        lines.append("")

    last_call = payload.get("last_call")
    if isinstance(last_call, dict):
        lines.append("— LAST CALL —")
        lines.append(
            f"  {last_call.get('call_type', '?')} at "
            f"{last_call.get('timestamp', '?')}  "
            f"in={int(last_call.get('prompt_tokens') or 0):,} "
            f"out={int(last_call.get('completion_tokens') or 0):,}  "
            f"cost={_fmt_money(last_call.get('cost_usd'))}"
        )

    return "\n".join(line for line in lines if line is not None).rstrip()


def _fmt_window(w: Mapping[str, Any]) -> str:
    calls = int(w.get("calls") or 0)
    prompt = int(w.get("prompt_tokens") or 0)
    cached = int(w.get("cached_tokens") or 0)
    completion = int(w.get("completion_tokens") or 0)
    cost = w.get("cost_usd")
    parts = [
        f"  calls:    {calls}",
        f"  prompt:   {prompt:,}" + (
            f"  (cached {cached:,})" if cached else ""
        ),
        f"  output:   {completion:,}",
        f"  cost:     {_fmt_money(cost)}",
    ]
    return "\n".join(parts)


def _fmt_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    if abs(n) >= 1.0:
        return f"${n:,.2f}"
    if abs(n) >= 0.01:
        return f"${n:.4f}"
    if n == 0:
        return "$0.00"
    return f"${n:.6f}"
