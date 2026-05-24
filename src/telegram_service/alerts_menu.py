"""Inline-keyboard rendering + callback parsing for the /alerts submenu.

Callback data format (Telegram caps at 64 bytes per button):

    alerts:toggle:<alert_type>      tap a row to flip ON/OFF
    alerts:close                    close/hide the menu

Keeping this isolated from :mod:`commands` keeps the dispatch table
slim and makes it trivial to unit-test the rendering in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_NAMESPACE = "alerts"
_PREFIX = f"{_NAMESPACE}:"

# Compact symbols so the keyboard fits on phone screens.
_ON_GLYPH = "ON  "
_OFF_GLYPH = "OFF "


@dataclass(frozen=True)
class ParsedAlertCallback:
    action: str        # "toggle" | "close" | "unknown"
    alert_type: str    # alert type for "toggle"; empty otherwise


def is_alerts_callback(data: str) -> bool:
    return isinstance(data, str) and data.startswith(_PREFIX)


def parse_alerts_callback(data: str) -> ParsedAlertCallback:
    """Parse the ``alerts:*`` callback_data emitted by our keyboards."""
    if not is_alerts_callback(data):
        return ParsedAlertCallback(action="unknown", alert_type="")
    parts = data.split(":", 2)
    # parts[0] == "alerts"
    if len(parts) < 2:
        return ParsedAlertCallback(action="unknown", alert_type="")
    action = parts[1]
    if action == "toggle":
        return ParsedAlertCallback(
            action="toggle",
            alert_type=parts[2] if len(parts) >= 3 else "",
        )
    if action == "close":
        return ParsedAlertCallback(action="close", alert_type="")
    return ParsedAlertCallback(action="unknown", alert_type="")


# ---------------------------------------------------------------------------
# Keyboard + caption builders
# ---------------------------------------------------------------------------

def build_alerts_keyboard(subscriptions_payload: dict[str, Any]) -> dict[str, Any]:
    """Build the Telegram ``reply_markup`` for the /alerts menu.

    ``subscriptions_payload`` is the body returned by the control API's
    ``GET /alerts/subscriptions?chat_id=X``, i.e. a dict containing an
    ``available`` list of ``{type, label, enabled}`` rows. We render one
    row per alert type plus a "Close" row at the bottom.
    """
    rows: list[list[dict[str, Any]]] = []
    for entry in subscriptions_payload.get("available") or []:
        type_str = str(entry.get("type") or "")
        label = str(entry.get("label") or type_str)
        enabled = bool(entry.get("enabled"))
        glyph = _ON_GLYPH if enabled else _OFF_GLYPH
        rows.append([{
            "text": f"{glyph}{label}",
            "callback_data": f"{_PREFIX}toggle:{type_str}",
        }])
    rows.append([{
        "text": "Close menu",
        "callback_data": f"{_PREFIX}close",
    }])
    return {"inline_keyboard": rows}


def build_alerts_caption(subscriptions_payload: dict[str, Any]) -> str:
    """One-line summary shown above the keyboard."""
    available = subscriptions_payload.get("available") or []
    enabled_count = sum(1 for e in available if e.get("enabled"))
    total = len(available)
    return (
        f"Telegram alert subscriptions ({enabled_count}/{total} enabled).\n"
        f"Tap a row to toggle. Tap 'Close menu' when you're done."
    )


def build_closed_caption(subscriptions_payload: dict[str, Any]) -> str:
    """Replacement caption shown after the user taps 'Close menu'."""
    available = subscriptions_payload.get("available") or []
    enabled_names = sorted(
        str(e.get("label") or e.get("type") or "")
        for e in available
        if e.get("enabled")
    )
    if not enabled_names:
        return "Telegram alerts: all OFF. Use /alerts to change."
    head = "Telegram alerts ON: " + ", ".join(enabled_names[:6])
    if len(enabled_names) > 6:
        head += f" (+{len(enabled_names) - 6} more)"
    return head + "\nUse /alerts to change."
