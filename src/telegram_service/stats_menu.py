"""Inline-keyboard rendering + callback parsing for /stats.

Callback data shape (Telegram caps at 64 bytes per button):

    stats:view:<view_id>     show a particular view
    stats:close              close/hide the menu

Available view IDs match :data:`STATS_VIEWS` keys below. Anything
unknown is rejected by the dispatcher with a quiet log line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_NAMESPACE = "stats"
_PREFIX = f"{_NAMESPACE}:"


# Display label, callback view id. The order here drives the keyboard
# layout. Kept short — Telegram inline buttons render on phone screens
# and long labels wrap badly.
STATS_VIEWS: tuple[tuple[str, str], ...] = (
    ("Overview",     "overview"),
    ("Today",        "today"),
    ("7 days",       "7d"),
    ("30 days",      "30d"),
    ("All-time",     "all"),
    ("Open",         "open"),
    ("Closed",       "closed"),
    ("By Symbol",    "by_symbol"),
    ("Daily",        "daily"),
)

VALID_VIEW_IDS = frozenset(vid for _label, vid in STATS_VIEWS)


@dataclass(frozen=True)
class ParsedStatsCallback:
    action: str        # "view" | "close" | "unknown"
    view_id: str       # populated for "view"; empty otherwise


def is_stats_callback(data: str) -> bool:
    return isinstance(data, str) and data.startswith(_PREFIX)


def parse_stats_callback(data: str) -> ParsedStatsCallback:
    if not is_stats_callback(data):
        return ParsedStatsCallback(action="unknown", view_id="")
    parts = data.split(":", 2)
    if len(parts) < 2:
        return ParsedStatsCallback(action="unknown", view_id="")
    action = parts[1]
    if action == "view":
        vid = parts[2] if len(parts) >= 3 else ""
        if vid not in VALID_VIEW_IDS:
            return ParsedStatsCallback(action="unknown", view_id="")
        return ParsedStatsCallback(action="view", view_id=vid)
    if action == "close":
        return ParsedStatsCallback(action="close", view_id="")
    return ParsedStatsCallback(action="unknown", view_id="")


def build_stats_keyboard() -> dict[str, Any]:
    """Build the inline keyboard for the /stats menu.

    Buttons are laid out in rows of three so most phones can show the
    whole grid without horizontal scrolling.
    """
    rows: list[list[dict[str, str]]] = []
    cur: list[dict[str, str]] = []
    for label, vid in STATS_VIEWS:
        cur.append({"text": label, "callback_data": f"{_PREFIX}view:{vid}"})
        if len(cur) == 3:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    rows.append([{"text": "Close", "callback_data": f"{_PREFIX}close"}])
    return {"inline_keyboard": rows}


def build_stats_caption() -> str:
    return (
        "[STATS]\n"
        "Tap a view to see the bot's performance.\n"
        "Each view shows Bot P/L (only positions the bot opened) and\n"
        "Account P/L (the whole account incl. your manual trades)."
    )
