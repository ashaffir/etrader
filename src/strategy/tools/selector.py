"""ToolSelector — pick the right subset of tools per (instrument, cycle).

Three layered signals drive the decision:

1. **Static rules**: asset class compatibility (``Tool.asset_classes``)
   and data availability (``requires_volume`` + ``ctx.has_volume``).
   This filter is cheap and runs first.

2. **Regime gating**: trending markets favor moving-average / cross
   tools; ranging markets favor mean-reversion (Bollinger, RSI).
   The :class:`~src.strategy.regime.MarketRegime` snapshot for the
   instrument decides which families get priority.

3. **Performance weighting**: every tool has a rolling hit-rate from
   :class:`ToolPerformanceLog`. Tools with persistently negative
   hit-rate (below ``min_hit_rate``) are demoted out of the run set
   unless an LLM is enabled (so the LLM can still see them as
   features). This caps damage from a tool that turns out to be
   non-predictive on a particular asset.

The selector returns a sorted list of tool names plus a brief
"reasoning trace" so the cycle log can show *why* a given tool was
included or skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .base import Tool, ToolContext


@dataclass(frozen=True)
class ToolSelectorConfig:
    """Tunables for the selector."""

    max_tools_per_cycle: int = 14
    min_hit_rate: float = 0.40        # below this rolling rate, demote
    min_observations: int = 30        # don't penalize a tool until N obs
    trending_priority: tuple[str, ...] = (
        "sma_cross", "ema_cross", "macd", "donchian", "trend_filter",
        "higher_tf_trend", "obv", "ad_line", "relative_strength",
    )
    ranging_priority: tuple[str, ...] = (
        "rsi", "bollinger", "cmf", "vwap", "volume_spike",
        "spread_filter", "market_hours",
    )
    always_include: tuple[str, ...] = (
        "spread_filter", "market_hours", "trend_filter", "instrument_feed",
    )


@dataclass(frozen=True)
class SelectionTrace:
    """Why each tool was included or skipped — surfaced in cycle logs."""

    selected: tuple[str, ...]
    skipped_static: tuple[tuple[str, str], ...]
    demoted_perf: tuple[tuple[str, float], ...]
    regime: str


class ToolSelector:
    """Stateless coordinator; reads from a registry + perf log per call."""

    def __init__(
        self,
        *,
        config: ToolSelectorConfig | None = None,
        performance_lookup: "PerfLookup | None" = None,
    ) -> None:
        self._cfg = config or ToolSelectorConfig()
        self._perf = performance_lookup

    def select(
        self,
        *,
        registry_tools: Iterable[Tool],
        ctx: ToolContext,
        regime: str = "trending",
    ) -> tuple[list[Tool], SelectionTrace]:
        """Return ``(tools_to_run, trace)`` for one (instrument, cycle).

        ``regime`` is one of ``"trending"``, ``"ranging"``, ``"unknown"``.
        """
        candidates: list[Tool] = []
        skipped: list[tuple[str, str]] = []
        for tool in registry_tools:
            if not tool.applies_to(ctx):
                skipped.append((tool.name, self._skip_reason(tool, ctx)))
                continue
            candidates.append(tool)

        priority = self._priority_for(regime)
        always = set(self._cfg.always_include)
        candidates.sort(key=lambda t: (
            0 if t.name in always else 1,
            priority.get(t.name, 99),
            t.name,
        ))

        demoted: list[tuple[str, float]] = []
        kept: list[Tool] = []
        for t in candidates:
            if len(kept) >= self._cfg.max_tools_per_cycle and t.name not in always:
                # Selector budget exhausted; surface why for traceability.
                skipped.append((t.name, "budget"))
                continue
            hit = self._hit_rate(t.name)
            if (
                hit is not None
                and hit < self._cfg.min_hit_rate
                and t.name not in always
            ):
                demoted.append((t.name, hit))
                continue
            kept.append(t)

        trace = SelectionTrace(
            selected=tuple(t.name for t in kept),
            skipped_static=tuple(skipped),
            demoted_perf=tuple(demoted),
            regime=regime,
        )
        return kept, trace

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _hit_rate(self, name: str) -> float | None:
        if self._perf is None:
            return None
        stats = self._perf.lookup(name)
        if stats is None:
            return None
        if stats.observations < self._cfg.min_observations:
            return None
        return stats.hit_rate

    def _priority_for(self, regime: str) -> dict[str, int]:
        """Return name → priority (lower = run earlier)."""
        if regime == "ranging":
            order = self._cfg.ranging_priority + self._cfg.trending_priority
        else:
            order = self._cfg.trending_priority + self._cfg.ranging_priority
        return {name: i for i, name in enumerate(order)}

    @staticmethod
    def _skip_reason(tool: Tool, ctx: ToolContext) -> str:
        if tool.requires_volume and not ctx.has_volume:
            return "no volume data"
        if ctx.asset_class not in tool.asset_classes:
            return f"asset_class {ctx.asset_class.value} not supported"
        return "n/a"


class PerfLookup:  # pragma: no cover - tiny shim, real impl in performance.py
    """Protocol — performance.py provides the real implementation.

    Kept as a duck-typed shim so ``selector.py`` doesn't import the
    performance module (avoids circular import; the runner injects).
    """

    def lookup(self, tool_name: str) -> "PerfStats | None":
        raise NotImplementedError


class PerfStats:  # pragma: no cover - dataclass placeholder; real one is in performance.py
    observations: int
    hit_rate: float
