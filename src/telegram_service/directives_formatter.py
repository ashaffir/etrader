"""Renderers for the /directives Telegram surface.

Two views:

* :func:`format_directives` — list current structured directives +
  notes + a short help footer telling the operator how to edit.
* :func:`format_set_result` / :func:`format_clear_result` /
  :func:`format_note_result` — one-line confirmations after an edit.

Kept text-only on purpose: the directive schema is small (5 fields)
so a plain text dump is faster to read than an inline keyboard.
"""

from __future__ import annotations

from typing import Any, Mapping


def format_directives(payload: Mapping[str, Any]) -> str:
    """Render the snapshot returned by :meth:`BotController.snapshot_directives`."""
    if not payload.get("enabled", True):
        return (
            "[DIRECTIVES]\n"
            "Directives store is not wired (server starting up?). "
            "Try again in a moment."
        )
    values = dict(payload.get("values") or {})
    lines = ["[DIRECTIVES]"]
    lines.append(
        f"no_overnight                : {_fmt_bool(values.get('no_overnight'))}"
    )
    lines.append(
        f"hold_ceiling_minutes        : "
        f"{_fmt_zero_disabled_int(values.get('hold_ceiling_minutes'))}"
    )
    lines.append(
        f"max_total_account_invested  : "
        f"{_fmt_zero_disabled_money(values.get('max_total_account_invested_usd'))}"
    )
    lines.append(
        f"pre_earnings_close_hours    : "
        f"{_fmt_zero_disabled_int(values.get('pre_earnings_close_hours'))}"
    )
    lines.append(
        f"blocked_symbols             : "
        f"{_fmt_list(values.get('blocked_symbols'))}"
    )
    lines.append(
        f"blocked_sectors             : "
        f"{_fmt_list(values.get('blocked_sectors'))}"
    )
    notes = str(values.get("notes") or "")
    if notes:
        lines.append("")
        lines.append("notes:")
        for line in notes.splitlines() or [notes]:
            lines.append(f"  {line}")
    else:
        lines.append("")
        lines.append("notes: (none)")
    lines.append("")
    lines.append("Edit with:")
    lines.append("  /directive set <key> <value>")
    lines.append("  /directive clear <key>")
    lines.append("  /note <add|set> <text>      (replaces current notes)")
    lines.append("  /note clear")
    lines.append("")
    lines.append("Examples:")
    lines.append("  /directive set no_overnight true")
    lines.append("  /directive set hold_ceiling_minutes 120")
    lines.append("  /directive set blocked_symbols NVDA,TSLA")
    lines.append("  /directive set max_total_account_invested_usd 3000")
    lines.append("  /directive set pre_earnings_close_hours 24")
    lines.append("  /note add Prefer financial-sector names this week.")
    return "\n".join(lines)


def format_set_result(payload: Mapping[str, Any]) -> str:
    return (
        f"Updated directive `{payload.get('key')}`\n"
        f"  before: {_fmt_value(payload.get('previous'))}\n"
        f"  after:  {_fmt_value(payload.get('current'))}"
    )


def format_clear_result(payload: Mapping[str, Any]) -> str:
    return (
        f"Cleared directive `{payload.get('key')}`\n"
        f"  was:        {_fmt_value(payload.get('previous'))}\n"
        f"  default:    {_fmt_value(payload.get('current'))}"
    )


def format_note_result(payload: Mapping[str, Any], *, cleared: bool = False) -> str:
    if cleared:
        prev = str(payload.get('previous') or '')
        return f"Notes cleared ({len(prev)} chars removed)."
    prev = str(payload.get('previous') or '')
    cur = str(payload.get('current') or '')
    return (
        f"Notes updated.\n"
        f"  before: {len(prev)} chars\n"
        f"  after:  {len(cur)} chars\n\n{cur}" if cur else
        f"Notes cleared ({len(prev)} chars removed)."
    )


# ---------------------------------------------------------------------------
# Tiny formatters
# ---------------------------------------------------------------------------

def _fmt_bool(v: Any) -> str:
    return "true" if bool(v) else "false"


def _fmt_zero_disabled_int(v: Any) -> str:
    try:
        n = int(v or 0)
    except (TypeError, ValueError):
        return "(disabled)"
    return f"{n} min" if n > 0 else "(disabled)"


def _fmt_zero_disabled_money(v: Any) -> str:
    try:
        n = float(v or 0)
    except (TypeError, ValueError):
        return "(disabled)"
    return f"${n:.2f}" if n > 0 else "(disabled)"


def _fmt_list(v: Any) -> str:
    if not v:
        return "(none)"
    if isinstance(v, (list, tuple)):
        items = [str(x) for x in v if str(x).strip()]
        return ", ".join(items) if items else "(none)"
    return str(v)


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return _fmt_bool(v)
    if isinstance(v, (list, tuple)):
        return _fmt_list(v)
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)
