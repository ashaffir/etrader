"""Telegram-side renderers for ``/channels`` and ``/channels test``.

Kept separate from :mod:`src.telegram_service.formatters` so neither
file grows past the project's 300-LOC limit. Pure functions, no I/O —
the control client is the one that talks to the bot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# /channels — overview
# ---------------------------------------------------------------------------

def format_channels_overview(payload: Mapping[str, Any]) -> str:
    """Render the per-source health overview (``GET /news/channels``)."""
    channels = list(payload.get("channels") or [])
    lines = ["[CHANNELS] news source overview"]

    pipeline_enabled = bool(payload.get("pipeline_enabled", True))
    scan_min = payload.get("scan_interval_minutes")
    ttl_h = payload.get("ttl_hours")
    half_h = payload.get("half_life_hours")
    lines.append(
        f"pipeline:  {'on' if pipeline_enabled else 'OFF'}  "
        f"scan_every={scan_min}m  ttl={ttl_h}h  half_life={half_h}h"
    )

    last_scan = payload.get("last_scan")
    if last_scan:
        lines.append(
            f"last scan: {_fmt_unix(last_scan.get('finished_at_unix'))}  "
            f"items_kept={last_scan.get('items_kept', 0)}  "
            f"obs={last_scan.get('observations_recorded', 0)}"
        )
    else:
        lines.append("last scan: (none yet)")

    next_in_s = payload.get("next_scan_in_seconds")
    if next_in_s is not None:
        mins = max(0, int(float(next_in_s) / 60))
        lines.append(f"next scan in: ~{mins} min")
    lines.append("")

    if not channels:
        lines.append("(no news sources are configured)")
        return "\n".join(lines)

    lines.append("name           cfg  wired  weight  last  status")
    for ch in channels:
        name = (str(ch.get("name") or "?"))[:13].ljust(13)
        enabled = "yes" if ch.get("enabled") else "no "
        wired = "yes" if ch.get("wired") else "no "
        weight = ch.get("weight")
        weight_str = f"{float(weight):4.2f}" if isinstance(weight, (int, float)) else " -- "
        last_items = int(ch.get("last_items_kept") or 0)
        last_str = f"{last_items:>4d}"
        status = _channel_status(ch)
        lines.append(
            f"{name}  {enabled}  {wired}    {weight_str}    {last_str}  {status}"
        )

    failing = [c for c in channels if c.get("disabled_reason") or c.get("last_error")]
    if failing:
        lines.append("")
        lines.append("Issues:")
        for ch in failing[:10]:
            lines.append(f"  {ch.get('name')}: {_issue_text(ch)}")

    lines.append("")
    lines.append("Probe live with `/channels test` or `/channels test <name1,name2>`.")
    return "\n".join(lines)


def _channel_status(ch: Mapping[str, Any]) -> str:
    if ch.get("disabled_reason"):
        return "DISABLED"
    if ch.get("last_error"):
        return "ERROR"
    if not ch.get("wired"):
        return "unknown"
    if not ch.get("enabled"):
        return "off"
    return "ok"


def _issue_text(ch: Mapping[str, Any]) -> str:
    reason = ch.get("disabled_reason")
    if reason:
        return f"disabled — {_truncate(str(reason), 120)}"
    err = ch.get("last_error")
    if err:
        return f"last error — {_truncate(str(err), 120)}"
    return "(no detail)"


# ---------------------------------------------------------------------------
# /channels test — dry run
# ---------------------------------------------------------------------------

def format_channels_test(payload: Mapping[str, Any]) -> str:
    """Render the dry-run probe result (``POST /news/channels/test``)."""
    if not payload.get("available", True):
        return (
            "[CHANNELS test]\n"
            "News pipeline isn't wired in this process — nothing to probe.\n"
            "(Is the trading bot running with `[news] enabled = true`?)"
        )
    summary = payload.get("summary") or {}
    results = list(payload.get("results") or [])
    lines = [
        "[CHANNELS test] live dry-run",
        (
            f"probed={summary.get('probed', 0)}  "
            f"ok={summary.get('ok', 0)}  "
            f"failed={summary.get('failed', 0)}"
        ),
        "",
    ]
    if not results:
        lines.append("(no sources matched the filter)")
        return "\n".join(lines)

    lines.append("name           ok    items  ms     detail")
    for r in results:
        name = (str(r.get("name") or "?"))[:13].ljust(13)
        ok = "OK " if r.get("ok") else "FAIL"
        items = int(r.get("items_count") or 0)
        ms = int(r.get("duration_ms") or 0)
        detail = _result_detail(r)
        lines.append(f"{name}  {ok}  {items:>5d}  {ms:>5d}  {detail}")

    failures = [r for r in results if not r.get("ok")]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for r in failures:
            lines.append(
                f"  {r.get('name')}: "
                f"{r.get('disabled_reason') or r.get('error') or '(no reason)'}"
            )
    return "\n".join(lines)


def _result_detail(r: Mapping[str, Any]) -> str:
    if r.get("disabled_reason"):
        return f"disabled: {_truncate(str(r['disabled_reason']), 60)}"
    if not r.get("ok"):
        return f"error: {_truncate(str(r.get('error') or '?'), 60)}"
    sample = r.get("sample_headline")
    if sample:
        return _truncate(str(sample), 60)
    return "(no items)"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _fmt_unix(ts: Any) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%H:%M:%S UTC")
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# argument parsing for `/channels [test] [logs] [name1,name2]`
# ---------------------------------------------------------------------------

def parse_channels_args(raw: str) -> tuple[str, list[str]]:
    """Split the ``/channels`` argument string into ``(subcommand, names)``.

    Recognised subcommands: ``""`` (overview), ``"test"``, ``"logs"``.
    Anything else (e.g. ``"stocktwits,yfinance"``) is treated as a name
    list under the implicit ``"test"`` subcommand — the operator typed
    a source name, the intent is "probe those".
    """
    text = (raw or "").strip()
    if not text:
        return "", []
    parts = text.split(maxsplit=1)
    head = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if head in {"test", "probe", "dry-run", "dryrun"}:
        return "test", _split_names(rest)
    if head in {"log", "logs", "errors"}:
        return "logs", _split_names(rest)
    return "test", _split_names(text)


def _split_names(blob: str) -> list[str]:
    out: list[str] = []
    for part in (blob or "").replace(";", ",").split(","):
        chunk = part.strip()
        if not chunk:
            continue
        for piece in chunk.split():
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


# ---------------------------------------------------------------------------
# /channels logs — recent per-source errors from the last scan
# ---------------------------------------------------------------------------

def format_channels_logs(payload: Mapping[str, Any]) -> str:
    """Render recent per-source errors / counts from the last scan."""
    last_scan = payload.get("last_scan")
    channels = list(payload.get("channels") or [])
    if not last_scan:
        return (
            "[CHANNELS logs]\n"
            "No scan has run yet — the aggregator fires every "
            f"{payload.get('scan_interval_minutes', '?')} minutes."
        )
    lines = ["[CHANNELS logs] most recent scan"]
    lines.append(
        f"started:  {_fmt_unix(last_scan.get('started_at_unix'))}"
    )
    lines.append(
        f"finished: {_fmt_unix(last_scan.get('finished_at_unix'))}"
    )
    lines.append(
        f"totals:   fetched={last_scan.get('items_fetched', 0)}  "
        f"kept={last_scan.get('items_kept', 0)}  "
        f"obs={last_scan.get('observations_recorded', 0)}"
    )
    lines.append("")
    lines.append("Per-source items kept:")
    counts = last_scan.get("per_source_counts") or {}
    if counts:
        for name, count in sorted(counts.items()):
            lines.append(f"  {name:<14} {int(count or 0):>4d}")
    else:
        lines.append("  (none recorded)")
    errors = last_scan.get("per_source_errors") or {}
    lines.append("")
    if errors:
        lines.append("Per-source errors:")
        for name, err in errors.items():
            lines.append(f"  {name}: {_truncate(str(err), 200)}")
    else:
        lines.append("Per-source errors: (none)")

    disabled = [c for c in channels if c.get("disabled_reason")]
    if disabled:
        lines.append("")
        lines.append("Configured-but-disabled sources:")
        for ch in disabled:
            lines.append(f"  {ch.get('name')}: {_truncate(str(ch['disabled_reason']), 200)}")
    return "\n".join(lines)


def channels_help_text() -> str:
    return (
        "Usage:\n"
        "  /channels                 source overview (status, weights, last counts)\n"
        "  /channels test            live dry-run against every configured source\n"
        "  /channels test <names>    probe a subset, e.g. `/channels test stocktwits,yfinance`\n"
        "  /channels logs            most recent scan stats + per-source errors\n"
    )


def format_channels_help() -> str:
    return "[CHANNELS]\n" + channels_help_text()


__all__ = [
    "channels_help_text",
    "format_channels_help",
    "format_channels_logs",
    "format_channels_overview",
    "format_channels_test",
    "parse_channels_args",
]


# Sequence type imported for type-checker visibility (unused at runtime).
_ = Sequence
