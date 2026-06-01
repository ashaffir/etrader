"""Glue between the cycle runner and the tool catalog.

Owns construction of the tool registry, selector, runner, and the
per-cycle bookkeeping (cross-asset regime fetch, daily-candle fetch,
instrument-meta lookup, performance log settle). The cycle module
just calls :meth:`ToolOrchestrator.evaluate_candidates` once per
cycle and gets back a dict mapping ``instrument_id`` →
:class:`ToolRunResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..config import AppConfig
from ..etoro.client import EtoroClient
from ..etoro.feeds import InstrumentFeedFetcher
from ..etoro.market_data import (
    Candle,
    InstrumentMeta,
    LiveRate,
    fetch_candles,
    search_instrument,
)
from .performance import ToolPerformanceLog
from .regime import (
    CrossAssetRegime,
    detect_cross_asset_regime,
    detect_instrument_regime,
)
from .signals import Candidate
from .tools.base import ToolContext, asset_class_for
from .tools.registry import ToolRegistry, register_default_tools
from .tools.runner import ToolRunResult, ToolRunner, render_trace
from .tools.selector import ToolSelector, ToolSelectorConfig


@dataclass
class CycleAnchorState:
    """Anchor candles + cross-asset regime, cached for the cycle."""

    spx_candles: list[Candle]
    btc_candles: list[Candle]
    regime: CrossAssetRegime


class ToolOrchestrator:
    """Build once per process; reused across cycles."""

    def __init__(
        self,
        *,
        cfg: AppConfig,
        etoro: EtoroClient,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._cfg = cfg
        self._etoro = etoro
        self._log = logger or logging.getLogger("etrader.strategy.tools.orchestrator")
        self._performance = ToolPerformanceLog(
            path=Path(cfg.tools.performance_log_path),
            logger=self._log,
        )
        self._feed_fetcher: InstrumentFeedFetcher | None = None
        if cfg.tools.feed_enabled:
            self._feed_fetcher = InstrumentFeedFetcher(
                etoro,
                take=cfg.tools.feed_take,
                cache_ttl_seconds=cfg.tools.feed_cache_ttl_seconds,
            )
        self._registry: ToolRegistry = register_default_tools(
            feed_fetcher=self._feed_fetcher,
            feed_enabled=cfg.tools.feed_enabled,
            spread_max_pct=cfg.tools.spread_max_pct,
        )
        selector_cfg = ToolSelectorConfig(
            max_tools_per_cycle=cfg.tools.max_tools_per_cycle,
            min_hit_rate=cfg.tools.min_hit_rate,
            min_observations=cfg.tools.min_observations,
        )
        self._selector = ToolSelector(
            config=selector_cfg,
            performance_lookup=self._performance,
        )
        self._runner = ToolRunner(
            registry=self._registry,
            selector=self._selector,
            logger=self._log,
        )
        self._anchor_ids: dict[str, int] = {}
        self._anchor_state: CycleAnchorState | None = None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def performance(self) -> ToolPerformanceLog:
        return self._performance

    def warmup_anchors(self) -> None:
        """Resolve regime anchor symbols → IDs once at startup."""
        for sym in self._cfg.tools.regime_anchors:
            try:
                inst_id = search_instrument(self._etoro, sym)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("[tools] regime anchor lookup failed for %s: %s", sym, exc)
                continue
            if inst_id:
                self._anchor_ids[sym] = inst_id

    def refresh_anchor_state(self) -> CrossAssetRegime | None:
        """Fetch cross-asset regime once per cycle and stash it."""
        if not self._anchor_ids:
            self._anchor_state = None
            return None
        spx = self._fetch_anchor("SPX500")
        btc = self._fetch_anchor("BTC")
        regime = detect_cross_asset_regime(spx_candles=spx, btc_candles=btc)
        self._anchor_state = CycleAnchorState(
            spx_candles=list(spx),
            btc_candles=list(btc),
            regime=regime,
        )
        return regime

    def evaluate_candidates(
        self,
        *,
        candidates: Sequence[Candidate],
        candles_by_instrument: Mapping[int, Sequence[Candle]],
        rates: Mapping[int, LiveRate],
        instrument_metas: Mapping[int, InstrumentMeta],
        higher_tf_candles: Mapping[int, Sequence[Candle]] | None = None,
        cycle_index: int = 0,
        earnings_lookup: Any | None = None,
    ) -> dict[int, ToolRunResult]:
        results: dict[int, ToolRunResult] = {}
        regime_dict = (
            self._anchor_state.regime.to_dict() if self._anchor_state else None
        )
        for cand in candidates:
            inst_id = cand.instrument_id
            candles = list(candles_by_instrument.get(inst_id) or [])
            if not candles:
                continue
            meta = instrument_metas.get(inst_id)
            asset_class = asset_class_for(meta, symbol=cand.symbol)
            ctx = ToolContext(
                instrument_id=inst_id,
                symbol=cand.symbol,
                asset_class=asset_class,
                candidate_action=cand.action,
                strategy=self._cfg.strategy,
                guardrails=self._cfg.guardrails,
                candles=candles,
                rate=rates.get(inst_id),
                instrument_meta=meta,
                higher_tf_candles=list((higher_tf_candles or {}).get(inst_id) or []),
                cross_asset_regime=regime_dict,
                earnings_lookup=earnings_lookup,
            )
            instrument_regime = detect_instrument_regime(candles).label
            run_result = self._runner.run(ctx=ctx, regime=instrument_regime)
            results[inst_id] = run_result
            if run_result.scores:
                self._performance.record_scores(
                    cycle_index=cycle_index,
                    instrument_id=inst_id,
                    scores=run_result.scores,
                )
            self._log.debug(
                "[tools] %s — %s",
                cand.symbol,
                render_trace(run_result.trace),
            )
        # Record cycle close prices so the perf log can settle on the next cycle.
        for inst_id, candles in candles_by_instrument.items():
            if not candles:
                continue
            self._performance.record_close(
                cycle_index=cycle_index,
                instrument_id=inst_id,
                close=float(candles[-1].close),
            )
        # Settle prior cycle's outcomes against current closes.
        self._performance.settle(current_cycle=cycle_index)
        return results

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _fetch_anchor(self, symbol: str) -> list[Candle]:
        inst_id = self._anchor_ids.get(symbol)
        if inst_id is None:
            return []
        try:
            return fetch_candles(
                self._etoro,
                inst_id,
                interval=self._cfg.tools.higher_tf_interval,
                count=self._cfg.tools.higher_tf_count,
            )
        except Exception as exc:  # noqa: BLE001 - regime is a soft signal
            self._log.warning(
                "[tools] anchor candle fetch failed for %s: %s", symbol, exc,
            )
            return []
