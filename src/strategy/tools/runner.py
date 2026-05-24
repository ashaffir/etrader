"""ToolRunner — executes the selected tools and aggregates results.

Given a candidate from the deterministic signal layer, the runner:

1. asks the :class:`ToolSelector` which subset to run for this
   (instrument, cycle, regime, asset class);
2. invokes each selected tool's ``evaluate``;
3. collects features, an aggregate +/- score, and any veto from a
   gate tool;
4. returns a single :class:`ToolRunResult` so the cycle code can pass
   it to the decision engine in one shot.

Tool evaluation errors are *isolated*: a single tool blowing up does
not poison the cycle. We log the failure, drop that tool's
contribution, and continue. This matches the rest of the bot's
"degrade, don't crash" posture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from .base import Tool, ToolContext, ToolResult
from .registry import ToolRegistry
from .selector import SelectionTrace, ToolSelector


@dataclass
class ToolRunResult:
    """Aggregated output of running every selected tool against one candidate."""

    instrument_id: int
    symbol: str
    features: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    selected_tools: tuple[str, ...] = ()
    gate_passed: bool = True
    gate_reason: str = ""
    errors: dict[str, str] = field(default_factory=dict)
    trace: SelectionTrace | None = None

    @property
    def aggregate_score(self) -> float:
        """Mean of per-tool scores; bounded -1..+1.

        Used by the deterministic fallback as a tiebreaker when the
        LLM is disabled.
        """
        vals = [s for s in self.scores.values() if s is not None]
        if not vals:
            return 0.0
        return max(-1.0, min(1.0, sum(vals) / len(vals)))


class ToolRunner:
    """Drive a registry against a single candidate."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        selector: ToolSelector,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._registry = registry
        self._selector = selector
        self._log = logger or logging.getLogger("etrader.strategy.tools.runner")

    def run(
        self,
        *,
        ctx: ToolContext,
        regime: str = "trending",
    ) -> ToolRunResult:
        selected, trace = self._selector.select(
            registry_tools=list(self._registry),
            ctx=ctx,
            regime=regime,
        )
        result = ToolRunResult(
            instrument_id=ctx.instrument_id,
            symbol=ctx.symbol,
            selected_tools=tuple(t.name for t in selected),
            trace=trace,
        )
        for tool in selected:
            try:
                outcome = tool.evaluate(ctx)
            except Exception as exc:  # noqa: BLE001 - any tool may raise; isolate it
                self._log.warning(
                    "[tools] %s failed for %s: %s",
                    tool.name, ctx.symbol, exc,
                )
                result.errors[tool.name] = str(exc)
                continue
            self._absorb(result, tool, outcome)
            if not result.gate_passed:
                # First gate failure short-circuits — there's no point
                # paying for the remaining tools when the candidate is
                # already vetoed.
                break
        return result

    @staticmethod
    def _absorb(result: ToolRunResult, tool: Tool, outcome: ToolResult) -> None:
        for key, value in (outcome.features or {}).items():
            # Namespace features by tool to keep the LLM prompt
            # unambiguous and to make per-tool feature lookup trivial.
            result.features[f"{tool.name}.{key}"] = value
        if outcome.score is not None:
            result.scores[tool.name] = float(outcome.score)
        if tool.role in ("gate", "both") and not outcome.gate_passed:
            result.gate_passed = False
            result.gate_reason = (
                f"{tool.name}: {outcome.gate_reason}"
                if outcome.gate_reason
                else tool.name
            )


def features_to_compact_dict(result: ToolRunResult) -> dict[str, Any]:
    """Project a ToolRunResult into the form the LLM prompt consumes."""
    return {
        "selected_tools": list(result.selected_tools),
        "features": dict(result.features),
        "scores": dict(result.scores),
        "aggregate_score": result.aggregate_score,
        "gate_passed": result.gate_passed,
        "gate_reason": result.gate_reason,
    }


def render_trace(trace: SelectionTrace | None) -> str:
    """One-line summary of which tools were used vs skipped, for cycle logs."""
    if trace is None:
        return ""
    parts = [f"regime={trace.regime}", f"selected={','.join(trace.selected) or '—'}"]
    if trace.skipped_static:
        skipped = ",".join(f"{n}({r})" for n, r in trace.skipped_static[:5])
        parts.append(f"skipped={skipped}")
    if trace.demoted_perf:
        demoted = ",".join(f"{n}({r:.2f})" for n, r in trace.demoted_perf[:5])
        parts.append(f"demoted={demoted}")
    return " | ".join(parts)
