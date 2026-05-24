"""Tool catalog: pluggable indicators that produce features and gates.

The bot used to evaluate a tiny fixed indicator set (SMA cross, RSI,
momentum). This package replaces that with a registry of named tools
the LLM can reason over, plus a selector that picks the relevant
subset per (instrument, cycle) using static asset-class rules,
real-time market regime, and rolling per-tool performance stats.

Public API:

- :class:`Tool` — base interface every tool implements.
- :class:`ToolContext` — per-(instrument, cycle) inputs.
- :class:`ToolResult` — feature value(s) and an optional gate verdict.
- :class:`ToolRegistry` — register/lookup tools by name.
- :class:`ToolRunner` — drive a registry against a candidate.
- :class:`ToolSelector` — pick the relevant subset.
- :func:`register_default_tools` — populate the registry with the
  built-in price/volume/context tools.
"""

from .base import (
    AssetClass,
    Tool,
    ToolContext,
    ToolResult,
    asset_class_for,
)
from .registry import ToolRegistry, register_default_tools
from .runner import ToolRunner, ToolRunResult
from .selector import ToolSelector, ToolSelectorConfig

__all__ = [
    "AssetClass",
    "Tool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "ToolRunner",
    "ToolRunResult",
    "ToolSelector",
    "ToolSelectorConfig",
    "asset_class_for",
    "register_default_tools",
]
