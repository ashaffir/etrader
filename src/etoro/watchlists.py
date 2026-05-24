"""Watchlists, curated lists, market recommendations — all read-only helpers.

These are entirely optional but the bot exposes them so the universe
builder can pull the user's existing watchlist as one possible source.
"""

from __future__ import annotations

from typing import Any

from .client import EtoroClient
from .errors import EtoroApiError


def fetch_default_watchlist_items(client: EtoroClient) -> list[dict[str, Any]]:
    try:
        payload = client.get("/watchlists/default-watchlist/selected-items", retries=1)
    except EtoroApiError:
        return []
    return list((payload or {}).get("items") or [])


def fetch_curated_lists(client: EtoroClient) -> list[dict[str, Any]]:
    try:
        payload = client.get("/curated-lists", retries=1)
    except EtoroApiError:
        return []
    return list((payload or {}).get("lists") or payload or [])


def fetch_market_recommendations(client: EtoroClient, items_count: int = 25) -> list[dict[str, Any]]:
    items_count = max(1, min(int(items_count), 100))
    try:
        payload = client.get(f"/market-recommendations/{items_count}", retries=1)
    except EtoroApiError:
        return []
    if isinstance(payload, list):
        return payload
    return list((payload or {}).get("items") or [])
