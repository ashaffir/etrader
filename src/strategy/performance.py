"""Per-tool performance log + rolling stats.

Tracks how predictive each tool has been, so the selector can demote
tools that are persistently wrong on a particular asset class. The
log is append-only JSONL on disk, and a small in-memory aggregator
maintains rolling hit-rate per tool.

Attribution is intentionally simple:

- when a tool emits a directional ``score`` for a candidate, we
  record ``(tool_name, instrument_id, sign(score), cycle_index)``;
- the next-cycle close return for that instrument is recorded too;
- a tool "hit" iff sign(score) matches sign(next_cycle_return).

This is a coarse proxy for predictive value but it's bias-free, runs
without any LLM evaluation, and gives the selector a useful signal
within tens of cycles. Real per-trade attribution (scoring tools by
realized P&L on closed positions) is left as a follow-up.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PerfStats:
    """Rolling counts maintained per tool name."""

    tool_name: str
    observations: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        if self.observations <= 0:
            return 0.0
        return self.hits / float(self.observations)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "observations": self.observations,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


class ToolPerformanceLog:
    """JSONL-backed performance log with a tiny in-memory rolling aggregator.

    Thread-safe (the cycle thread writes; the Telegram/HTTP thread can
    read stats). The on-disk file rotates by line count, not size:
    we only need a few thousand recent samples.
    """

    def __init__(
        self,
        *,
        path: Path,
        rolling_window: int = 500,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._path = Path(path)
        self._rolling_window = max(50, int(rolling_window))
        self._log = logger or logging.getLogger("etrader.strategy.performance")
        self._lock = threading.RLock()
        self._rolling: dict[str, deque[bool]] = {}
        self._scores: dict[int, dict[str, float]] = {}  # cycle_index -> tool_name -> score
        self._closes: dict[int, dict[int, float]] = {}  # cycle_index -> instrument_id -> close
        self._instrument_for: dict[int, dict[str, int]] = {}  # cycle_index -> tool_name -> instrument_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load_recent()

    # ------------------------------------------------------------------
    # Recording API (called by cycle.py)
    # ------------------------------------------------------------------

    def record_scores(
        self,
        *,
        cycle_index: int,
        instrument_id: int,
        scores: Mapping[str, float],
    ) -> None:
        """Stash this cycle's scores; outcomes are paired in the next call."""
        with self._lock:
            cur = self._scores.setdefault(cycle_index, {})
            inst_map = self._instrument_for.setdefault(cycle_index, {})
            for name, score in scores.items():
                key = f"{instrument_id}:{name}"
                cur[key] = float(score)
                inst_map[key] = instrument_id

    def record_close(
        self,
        *,
        cycle_index: int,
        instrument_id: int,
        close: float,
    ) -> None:
        with self._lock:
            self._closes.setdefault(cycle_index, {})[instrument_id] = float(close)

    def settle(self, *, current_cycle: int) -> None:
        """Pair scores from cycle ``current_cycle - 1`` with closes from this cycle.

        Called once per cycle, after the cycle's close prices are
        known. Resolves outcomes, writes append-only JSONL, and
        updates the rolling per-tool counts.
        """
        prior = current_cycle - 1
        with self._lock:
            scores = self._scores.pop(prior, None)
            inst_map = self._instrument_for.pop(prior, {})
            self._scores.pop(prior - 1, None)  # ditch one-cycle stale buckets
            self._instrument_for.pop(prior - 1, None)
            prior_closes = self._closes.pop(prior, {})
            cur_closes = self._closes.get(current_cycle, {})
            self._closes.pop(prior - 1, None)
            if not scores:
                return
            outcomes: list[dict[str, object]] = []
            for key, score in scores.items():
                inst = inst_map.get(key)
                if inst is None:
                    continue
                tool = key.split(":", 1)[1]
                p_close = prior_closes.get(inst)
                c_close = cur_closes.get(inst)
                if p_close is None or c_close is None or p_close <= 0:
                    continue
                ret = (c_close - p_close) / p_close
                hit = (score > 0 and ret > 0) or (score < 0 and ret < 0)
                self._bump(tool, hit)
                outcomes.append({
                    "ts": time.time(),
                    "cycle": prior,
                    "tool": tool,
                    "instrument_id": inst,
                    "score": round(score, 4),
                    "return_pct": round(ret * 100.0, 4),
                    "hit": hit,
                })
            if outcomes:
                self._append(outcomes)

    # ------------------------------------------------------------------
    # Stats API (consumed by the selector)
    # ------------------------------------------------------------------

    def stats_for(self, tool_name: str) -> PerfStats:
        with self._lock:
            window = self._rolling.get(tool_name)
            if not window:
                return PerfStats(tool_name=tool_name, observations=0, hits=0, misses=0)
            obs = len(window)
            hits = sum(1 for h in window if h)
            return PerfStats(
                tool_name=tool_name,
                observations=obs,
                hits=hits,
                misses=obs - hits,
            )

    def lookup(self, tool_name: str) -> PerfStats:
        """Selector-facing alias for ``stats_for`` (PerfLookup protocol)."""
        return self.stats_for(tool_name)

    def all_stats(self) -> list[PerfStats]:
        with self._lock:
            return [self.stats_for(name) for name in self._rolling]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _bump(self, tool: str, hit: bool) -> None:
        window = self._rolling.setdefault(tool, deque(maxlen=self._rolling_window))
        window.append(bool(hit))

    def _append(self, outcomes: Iterable[dict[str, object]]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for entry in outcomes:
                    f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            self._log.warning("[perf] could not write %s: %s", self._path, exc)

    def _load_recent(self) -> None:
        if not self._path.exists():
            return
        try:
            size = self._path.stat().st_size
            with self._path.open("rb") as f:
                if size > 1_000_000:  # only tail the last ~1 MB
                    f.seek(-1_000_000, os.SEEK_END)
                    f.readline()
                for raw in f:
                    try:
                        entry = json.loads(raw)
                    except (ValueError, json.JSONDecodeError):
                        continue
                    tool = entry.get("tool")
                    hit = bool(entry.get("hit"))
                    if isinstance(tool, str):
                        self._bump(tool, hit)
        except OSError as exc:
            self._log.warning("[perf] could not read %s: %s", self._path, exc)
