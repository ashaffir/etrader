"""Tool registry — maps name → Tool instance.

We keep a single, rebuilt-per-process registry so the cycle runner
gets a deterministic catalog. ``register_default_tools`` wires up
every built-in tool from the ``tools.*_tools`` modules; tests can
build a smaller registry by passing an explicit list.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .base import Tool


class ToolRegistry:
    """Append-and-lookup registry. Names must be unique."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"tool {type(tool).__name__} has no name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def by_family(self, family: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.family == family]

    def gates(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.role in ("gate", "both")]

    def features(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.role in ("feature", "both")]

    def descriptions(self) -> list[dict[str, object]]:
        return [
            {
                "name": t.name,
                "family": t.family,
                "role": t.role,
                "purpose": t.purpose,
                "asset_classes": [a.value for a in t.asset_classes],
                "requires_volume": t.requires_volume,
            }
            for t in self._tools.values()
        ]


def register_default_tools(
    *,
    feed_fetcher: object | None = None,
    extras: Iterable[Tool] | None = None,
    spread_max_pct: float | None = None,
    feed_enabled: bool = True,
) -> ToolRegistry:
    """Build the canonical registry with every built-in tool.

    ``feed_fetcher`` is injected because the instrument-feed tool
    needs an HTTP-bound callable; tests that don't want network
    activity can pass ``None``. ``spread_max_pct`` overrides the
    default cap on spread_filter; ``feed_enabled=False`` skips
    registering the feed tool entirely.
    """
    # Imports here to avoid a circular import at package init time.
    from . import context_tools, price_tools, volume_tools
    from .feed_tool import InstrumentFeedTool

    reg = ToolRegistry()
    for tool in price_tools.build_tools():
        reg.register(tool)
    for tool in volume_tools.build_tools():
        reg.register(tool)
    for tool in context_tools.build_tools():
        if spread_max_pct is not None and tool.name == "spread_filter":
            tool.max_spread_pct = float(spread_max_pct)  # type: ignore[attr-defined]
        reg.register(tool)
    if feed_enabled:
        reg.register(InstrumentFeedTool(fetcher=feed_fetcher))
    if extras:
        for t in extras:
            reg.register(t)
    return reg
