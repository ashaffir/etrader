"""One full trading cycle: fetch → decide → risk-gate → execute → monitor.

Pulled out of ``src.main`` so the entry-point file stays a thin wrapper
and the loop logic is independently testable. The cycle is deliberately
linear; readability beats cleverness when something goes wrong at 3 AM.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .ai.azure_client import AzureFoundryClient
from .ai.decision_context import (
    build_performance_block,
    project_bot_owned_positions,
    project_by_symbol_history,
)
from .alerts import AlertHub, AlertType
from .config import AppConfig
from .etoro.client import EtoroClient
from .etoro.market_data import (
    Candle,
    InstrumentMeta,
    LiveRate,
    fetch_candles,
    fetch_instruments,
    fetch_rates,
)
from .etoro.trading import (
    PortfolioSnapshot,
    compute_account_summary,
    fetch_portfolio,
)
from .execution.dynamic_stops import DynamicStopsStore
from .execution.executor import ExecutionResult, TradeExecutor
from .execution.monitor import PositionMonitor
from .execution.stuck_orders import CancelResult, StuckOrder
from .performance import PerformanceTracker
from .strategy.directive_enforcer import (
    build_directive_close_requests,
    prescreen_candidates,
)
from .strategy.directives import Directives, DirectivesStore
from .strategy.position_review import PositionReviewer
from .strategy.tools.base import AssetClass, asset_class_for
from .fundamentals import FundamentalsCache
from .news.scheduler import NewsScheduler
from .state import BotState
from .strategy.autotune import AutotuneState
from .strategy.autotune_parse import render_tune_diff
from .strategy.decisions import DecisionEngine, DecisionResult, render_decisions
from .strategy.ensemble import evaluate_ensemble
from .strategy.risk import RiskEvaluator, aggregate_summary
from .strategy.rules_summary import build_rules_payload
from .strategy.signals import build_candidates
from .strategy.tool_orchestration import ToolOrchestrator
from .strategy.tools.runner import ToolRunResult
from .strategy.universe import TrackedUniverse, UniverseBuilder
from .telemetry import TelemetryStore
from .trade_history import TradeHistoryEntry, TradeHistoryLog, utc_now_iso


@dataclass
class CycleContext:
    universe: TrackedUniverse
    last_universe_refresh_monotonic: float
    rate_cache: dict[int, LiveRate] = None  # type: ignore[assignment]
    instrument_metas: dict[int, "InstrumentMeta"] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rate_cache is None:
            self.rate_cache = {}
        if self.instrument_metas is None:
            self.instrument_metas = {}


class CycleRunner:
    """Owns one cycle's worth of work; instances are reused across cycles."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        etoro: EtoroClient,
        ai_client: AzureFoundryClient | None,
        state: BotState,
        universe_builder: UniverseBuilder,
        risk: RiskEvaluator,
        executor: TradeExecutor,
        monitor: PositionMonitor,
        decision_engine: DecisionEngine,
        log: logging.Logger | logging.LoggerAdapter,
        telemetry: TelemetryStore | None = None,
        history: TradeHistoryLog | None = None,
        tool_orchestrator: ToolOrchestrator | None = None,
        alerts: AlertHub | None = None,
        news_scheduler: NewsScheduler | None = None,
        fundamentals_cache: FundamentalsCache | None = None,
        autotune_state: AutotuneState | None = None,
        performance: PerformanceTracker | None = None,
        dynamic_stops: "DynamicStopsStore | None" = None,
        position_reviewer: "PositionReviewer | None" = None,
        directives_store: "DirectivesStore | None" = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._cfg = cfg
        self._etoro = etoro
        self._ai = ai_client
        self._state = state
        self._universe_builder = universe_builder
        self._risk = risk
        self._executor = executor
        self._monitor = monitor
        self._decision_engine = decision_engine
        self._log = log
        self._telemetry = telemetry
        self._history = history
        self._tools = tool_orchestrator
        self._alerts = alerts
        self._news_scheduler = news_scheduler
        # Optional Phase-3 fundamentals cache. When present we top it
        # up on every universe refresh and project a trim dict into
        # the LLM decision prompt for each candidate.
        self._fundamentals = fundamentals_cache
        # Autonomous-tuner overlay. When present, every cycle:
        # 1. raw_score histogram is folded in via observe_cycle();
        # 2. evidence digest is passed to the LLM decision call;
        # 3. any tuning block in the response is applied + alerted.
        self._autotune = autotune_state
        # Optional performance tracker. When present, the cycle records
        # every bot-attributable open/close + per-cycle mark-to-market
        # so /stats and the /ask LLM have authoritative P/L numbers.
        self._performance = performance
        # Per-position SL/TP override store. The LLM writes to it via
        # MODIFY_STOPS actions; the monitor reads from it on every
        # cycle to apply per-position bands (with trailing logic).
        self._dynamic_stops = dynamic_stops
        # Threshold-triggered reviewer. Flags open positions that
        # breached drawdown / pullback / stale-hold / max-hold so the
        # decision LLM is forced to attend to them this cycle.
        self._position_reviewer = position_reviewer
        # Persistent operator directives (hybrid structured + free-text
        # notes). The cycle prescreens candidates and emits directive-
        # driven CLOSEs (no_overnight / hold_ceiling); the risk
        # evaluator enforces hard structured rules; the LLM prompt
        # surfaces them so the manager honours both hard and soft
        # parts. ``None`` keeps every directive disabled.
        self._directives_store = directives_store
        # Optional stop signal so synchronous I/O in the cycle can be
        # short-circuited mid-flight. Set by the SIGINT/SIGTERM handler.
        self._stop_event = stop_event
        # State-change tracking so we only emit AI_UNAVAILABLE / DAILY_LOSS_HALT
        # alerts on edges, not every cycle.
        self._last_ai_available: bool | None = None
        self._last_halted: bool = False
        self._last_universe_symbols: tuple[str, ...] = ()

    def initial_universe(self) -> CycleContext:
        # First news scan happens before the first universe build so the
        # candidate store is warm. Forced run on boot — operators expect
        # to see real data on cycle 1, not "wait an hour for the scan".
        if self._news_scheduler is not None:
            self._log.info("[news] initial scan starting…")
            self._news_scheduler.maybe_run(force=True)
        universe = self._universe_builder.build(must_include={})
        ctx = CycleContext(
            universe=universe,
            last_universe_refresh_monotonic=time.monotonic(),
        )
        self._log.info("[universe] initial → %s", universe.summary_line())
        self._cache_instrument_metadata(universe, ctx.instrument_metas)
        self._refresh_fundamentals(universe)
        self._publish_universe(universe)
        # Seed the diff baseline so the first refresh isn't reported as a
        # "universe changed" alert.
        self._last_universe_symbols = tuple(
            sorted(universe.symbol_for_id.get(i, str(i)) for i in universe.instrument_ids)
        )
        if self._tools is not None:
            self._tools.warmup_anchors()
        return ctx

    def run_one(self, ctx: CycleContext) -> None:
        self._log.info(
            "─── cycle %d — %s ───",
            self._state.cycle_count,
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if self._telemetry is not None:
            self._telemetry.mark_cycle_started(self._state.cycle_count)
        self._maybe_run_news_scan(ctx)
        if self._abort_if_stopping("after news scan"):
            return
        self._maybe_refresh_universe(ctx)
        if self._abort_if_stopping("after universe refresh"):
            return

        rates = self._fetch_rates(ctx.universe, prior=ctx.rate_cache)
        ctx.rate_cache.update(rates)
        snapshot = self._fetch_portfolio()
        adopted = self._monitor.reconcile(snapshot, self._state)
        self._record_performance_opens(adopted, ctx)
        self._cancel_stuck_orders(snapshot)
        self._record_performance_vanished(snapshot)
        bot_owned_positions = [
            p for p in snapshot.positions
            if p.position_id in self._state.bot_owned_positions and p.mirror_id == 0
        ]
        summary = compute_account_summary(snapshot)
        self._record_performance_observe(bot_owned_positions, summary=summary)
        self._update_owned_instrument_ids(ctx, bot_owned_positions)
        self._publish_portfolio(snapshot, bot_owned_position_ids=[
            p.position_id for p in bot_owned_positions
        ], symbol_for_id=ctx.universe.symbol_for_id)
        if self._abort_if_stopping("after portfolio fetch"):
            return

        candles_by_instrument = self._fetch_candles(ctx.universe)
        candidates = build_candidates(
            cfg=self._cfg.strategy,
            candles_by_instrument=candles_by_instrument,
            symbol_for_id=ctx.universe.symbol_for_id,
            bot_owned_instrument_ids={p.instrument_id for p in bot_owned_positions},
        )

        directives = self._directives()
        if directives.blocked_symbols or directives.blocked_sectors:
            kept, dropped = prescreen_candidates(
                directives=directives,
                candidates=candidates,
                fundamentals_lookup=(
                    self._fundamentals.get if self._fundamentals is not None else None
                ),
            )
            if dropped:
                self._log.info(
                    "[directives] dropped %d candidate(s): %s",
                    len(dropped),
                    ", ".join(f"{s}({r})" for s, r in dropped),
                )
            candidates = kept

        self._log.info(
            "[signals] %d candidate(s): %s",
            len(candidates),
            ", ".join(f"{c.action} {c.symbol}({c.strength:.2f})" for c in candidates) or "—",
        )

        # Fold the *full* raw_score distribution (every tracked symbol,
        # not just the ones above threshold) into the autotuner state
        # BEFORE the LLM call so the evidence digest includes this
        # cycle's data. record_trades_placed below finalises the
        # snapshot once execution is done.
        if self._autotune is not None:
            self._autotune.observe_cycle(
                cycle_index=self._state.cycle_count,
                tracked_count=len(ctx.universe.instrument_ids),
                raw_scores=self._raw_scores_for_universe(candles_by_instrument),
                candidates_count=len(candidates),
            )

        tool_results: dict[int, ToolRunResult] = {}
        cross_asset_regime: dict | None = None
        if self._tools is not None and candidates:
            regime = self._tools.refresh_anchor_state()
            if regime is not None:
                cross_asset_regime = regime.to_dict()
                self._log.info("[regime] %s", regime.detail)
            tool_results = self._tools.evaluate_candidates(
                candidates=candidates,
                candles_by_instrument=candles_by_instrument,
                rates=rates,
                instrument_metas=ctx.instrument_metas,
                cycle_index=self._state.cycle_count,
            )
            gated = [
                (cand.symbol, tool_results[cand.instrument_id].gate_reason)
                for cand in candidates
                if cand.instrument_id in tool_results
                and not tool_results[cand.instrument_id].gate_passed
            ]
            if gated:
                self._log.info(
                    "[tools] %d gated: %s",
                    len(gated),
                    ", ".join(f"{s}({r})" for s, r in gated),
                )

        # summary computed above near reconcile; reuse here.
        rules_payload = build_rules_payload(
            strategy=self._cfg.strategy,
            guardrails=self._cfg.guardrails,
            tools=[],
        )
        if self._abort_if_stopping("before LLM decision"):
            return
        fundamentals_payload = self._fundamentals_for_candidates(candidates)
        autotune_evidence = self._build_autotune_evidence(snapshot)
        perf_summary = self._performance.summary() if self._performance else None
        reviews = self._run_position_review(
            bot_owned_positions, ctx.universe.symbol_for_id, ctx.rate_cache,
        )
        reviews_by_pos = {r.position_id: r for r in reviews}
        open_states = (
            self._performance.open_states_by_position()
            if self._performance else {}
        )
        enriched_owned = project_bot_owned_positions(
            positions=bot_owned_positions,
            symbol_for_id=ctx.universe.symbol_for_id,
            open_states=open_states,
            dynamic_stops=self._dynamic_stops,
            reviews_by_position_id=reviews_by_pos,
        )
        by_symbol_proj = self._build_by_symbol_projection(candidates, bot_owned_positions, ctx)
        performance_block = build_performance_block(
            summary=perf_summary,
            reviews=reviews,
            by_symbol_projection=by_symbol_proj,
        )
        position_units = {
            p.position_id: float(getattr(p, "units", 0.0) or 0.0)
            for p in bot_owned_positions
        }
        decision = self._decision_engine.decide(
            candidates=candidates,
            portfolio_summary=summary,
            bot_owned_positions=bot_owned_positions,
            symbol_for_id=ctx.universe.symbol_for_id,
            tool_results=tool_results,
            cross_asset_regime=cross_asset_regime,
            strategy_rules=rules_payload,
            fundamentals_by_symbol=fundamentals_payload,
            autotune_evidence=autotune_evidence,
            performance=performance_block,
            enriched_owned_positions=enriched_owned,
            position_units_by_id=position_units,
            directives=directives.to_dict(),
        )
        latency_str = f", {decision.latency_ms} ms" if decision.latency_ms is not None else ""
        self._log.info(
            "[ai] decision (%s%s): %s",
            "llm" if decision.llm_used else "deterministic",
            latency_str,
            render_decisions(decision.requests),
        )
        if decision.summary:
            self._log.debug("[ai] summary: %s", decision.summary)
        self._publish_decision(decision)
        # Edge-detect Azure availability AFTER the decision is published
        # so the alert reflects the cycle the user just saw.
        self.emit_ai_state_if_changed(decision_used_llm=decision.llm_used)
        # Apply any tuning the manager LLM emitted this cycle. The
        # changes take effect NEXT cycle's candidacy step (this cycle
        # has already past the build_candidates call) which is fine
        # for thresholds and weights; it's the intended cadence.
        self._apply_autotune(decision)

        bot_invested_total = sum(p.amount for p in bot_owned_positions)
        account_invested_total = float(summary.get("total_invested") or 0.0)
        directive_closes, directive_close_notes = build_directive_close_requests(
            directives=directives,
            bot_owned_positions=bot_owned_positions,
            symbol_for_id=ctx.universe.symbol_for_id,
            instrument_metas=ctx.instrument_metas,
            open_states=(
                self._performance.open_states_by_position()
                if self._performance else {}
            ),
            now=datetime.now(timezone.utc),
        )
        # Drop any LLM-emitted action that targets a position we're
        # already directive-closing (so we don't double-close and
        # never emit BUY for instruments whose existing position is
        # being flattened).
        if directive_close_notes:
            forced_pids = {int(n["position_id"]) for n in directive_close_notes}
            forced_inst = {int(n["instrument_id"]) for n in directive_close_notes}
            kept_requests = [
                r for r in decision.requests
                if not (
                    (r.position_id is not None and int(r.position_id) in forced_pids)
                    or (r.action == "BUY" and int(r.instrument_id) in forced_inst)
                )
            ]
            self._log.info(
                "[directives] auto-CLOSE %d position(s): %s",
                len(directive_close_notes),
                ", ".join(
                    f"{n['symbol']}({n['directive']})" for n in directive_close_notes
                ),
            )
            all_requests = kept_requests + directive_closes
        else:
            all_requests = list(decision.requests)

        verdicts = self._risk.evaluate(
            requests=all_requests,
            state=self._state,
            current_equity=summary.get("equity"),
            bot_owned_position_count=len(bot_owned_positions),
            bot_invested_total_usd=bot_invested_total,
            account_invested_total_usd=account_invested_total,
        )
        # Risk evaluation flips state.halted_today; capture the edge and
        # alert the operator the FIRST time we trip the kill switch today.
        self.emit_kill_switch_if_changed(halted=self._state.halted_today)
        agg = aggregate_summary(verdicts)
        denied_msgs = [v.reason for v in verdicts if not v.approved]
        self._log.info(
            "[risk] approved %d / %d — %s",
            agg["approved"], len(verdicts),
            ", ".join(denied_msgs) if denied_msgs else "all clear",
        )

        if self._abort_if_stopping("before order execution"):
            return
        results = self._executor.execute_all(
            verdicts=verdicts, rates=rates, state=self._state,
        )
        self._track_placed_orders(results, ctx)
        self._record_history(results)
        # Finalise this cycle's autotune snapshot with the actual
        # number of successful orders (BUY or CLOSE). Failed/skipped
        # status entries don't count: the manager needs to see when
        # the bot *actually* trades, not when the LLM intended to.
        if self._autotune is not None:
            actual_trades = sum(1 for r in results if r.status == "ok")
            self._autotune.record_trades_placed(trades_placed=actual_trades)

        self._log_portfolio_summary(summary, len(bot_owned_positions))
        self._log_results(results)
        if self._telemetry is not None:
            self._telemetry.mark_cycle_finished()

    # ------------------------------------------------------------------

    def _abort_if_stopping(self, where: str) -> bool:
        """Return True (and log) if a shutdown has been requested.

        Cycle phases call this between long-running synchronous steps so
        Ctrl-C feels responsive even mid-cycle. We never abort *during*
        order execution — only before — to avoid leaving the broker in
        an inconsistent state.
        """
        if self._stop_event is not None and self._stop_event.is_set():
            self._log.info("[cycle] stop requested — aborting %s", where)
            if self._telemetry is not None:
                self._telemetry.mark_cycle_finished()
            return True
        return False

    def _maybe_run_news_scan(self, ctx: CycleContext) -> None:
        """Let the news scheduler decide whether a scan is due.

        ``known_symbols`` is the current universe + any pending owned-
        position symbols, so per-ticker sources (yfinance, Yahoo RSS)
        always enrich what we already care about.
        """
        if self._news_scheduler is None:
            return
        known = set(ctx.universe.symbol_for_id.values())
        known.update(self._state.bot_owned_instrument_ids.values())
        self._news_scheduler.maybe_run(known_symbols=known)

    def _maybe_refresh_universe(self, ctx: CycleContext) -> None:
        elapsed_min = (time.monotonic() - ctx.last_universe_refresh_monotonic) / 60.0
        if elapsed_min < self._cfg.operations.universe_refresh_minutes:
            return
        self._log.info("[universe] refreshing (after %.1f min)", elapsed_min)
        ctx.universe = self._universe_builder.build(
            must_include=dict(self._state.bot_owned_instrument_ids),
        )
        ctx.last_universe_refresh_monotonic = time.monotonic()
        self._log.info("[universe] refreshed → %s", ctx.universe.summary_line())
        self._cache_instrument_metadata(ctx.universe, ctx.instrument_metas)
        self._refresh_fundamentals(ctx.universe)
        self._publish_universe(ctx.universe)
        self.emit_universe_change(ctx.universe)

    def _fundamentals_for_candidates(
        self,
        candidates: Iterable,
    ) -> dict[str, dict]:
        """Return ``{SYMBOL: trim_fundamentals_dict}`` for prompt enrichment.

        Returns an empty dict when:
        - the cache isn't wired,
        - ``[fundamentals] enrich_decision_prompt = false``, or
        - the universe is empty.
        """
        if self._fundamentals is None:
            return {}
        if not getattr(self._cfg.fundamentals, "enrich_decision_prompt", True):
            return {}
        from .fundamentals.prompt import project_for_llm  # local: avoid heavy imports at module load

        out: dict[str, dict] = {}
        for c in candidates:
            sym = (getattr(c, "symbol", "") or "").upper()
            if not sym:
                continue
            snap = self._fundamentals.get(sym)
            if snap is None:
                continue
            out[sym] = project_for_llm(snap)
        return out

    def _refresh_fundamentals(self, universe: TrackedUniverse) -> None:
        """Top up the fundamentals cache for the currently tracked symbols.

        Only stale entries are re-fetched (see
        :meth:`FundamentalsCache.is_stale`); refreshing is capped at
        ``budget_per_refresh`` so the cycle never blocks on a long
        chain of yfinance calls. The stop event is honoured between
        symbols so Ctrl-C bails out quickly.
        """
        if self._fundamentals is None:
            return
        symbols = [universe.symbol_for_id.get(i, "") for i in universe.instrument_ids]
        symbols = [s for s in symbols if s]
        if not symbols:
            return
        is_stopping = (
            self._stop_event.is_set if self._stop_event is not None else None
        )
        results = self._fundamentals.refresh(
            symbols,
            budget=int(self._cfg.fundamentals.budget_per_refresh),
            is_stopping=is_stopping,
        )
        refreshed = sum(1 for v in results.values() if v == "refreshed")
        failed = sum(1 for v in results.values() if v == "failed")
        skipped = sum(1 for v in results.values() if v == "skipped")
        if refreshed or failed or skipped:
            self._log.info(
                "[fundamentals] refreshed=%d failed=%d skipped=%d (cache size %d)",
                refreshed, failed, skipped, len(self._fundamentals),
            )

    def _update_owned_instrument_ids(
        self,
        ctx: CycleContext,
        bot_owned_positions: Iterable,
    ) -> None:
        """Refresh ``state.bot_owned_instrument_ids`` from the latest snapshot.

        Called after every portfolio reconcile so the next universe
        refresh knows which instruments must remain tracked. If a newly-
        opened position's instrument isn't currently tracked, force the
        next ``_maybe_refresh_universe`` call to fire on the next cycle
        (instead of waiting up to ``universe_refresh_minutes``).
        """
        symbol_for_id = ctx.universe.symbol_for_id
        owned: dict[int, str] = {}
        for p in bot_owned_positions:
            inst_id = int(getattr(p, "instrument_id", 0) or 0)
            if not inst_id:
                continue
            owned[inst_id] = symbol_for_id.get(inst_id, f"INST-{inst_id}")
        previous_keys = set(self._state.bot_owned_instrument_ids.keys())
        self._state.bot_owned_instrument_ids = owned

        unseen = [i for i in owned if i not in symbol_for_id]
        if unseen and set(unseen) - previous_keys:
            self._log.info(
                "[universe] %d owned instrument(s) not in tracked set "
                "(%s) — forcing universe refresh on next cycle",
                len(unseen),
                ", ".join(str(i) for i in unseen[:5]),
            )
            # Backdate the refresh timer so _maybe_refresh_universe
            # fires on the next cycle entry; no need to bypass the
            # rate/candle work already done this cycle.
            ctx.last_universe_refresh_monotonic = 0.0

    def _cache_instrument_metadata(
        self,
        universe: TrackedUniverse,
        sink: dict[int, InstrumentMeta],
    ) -> None:
        if not universe.instrument_ids:
            return
        try:
            metas = fetch_instruments(self._etoro, universe.instrument_ids)
        except Exception as exc:  # noqa: BLE001 — non-fatal enrichment
            self._log.warning("[universe] instrument metadata fetch failed: %s", exc)
            return
        for inst_id, meta in metas.items():
            sink[inst_id] = meta
            if meta.symbol_full and inst_id not in universe.symbol_for_id:
                universe.symbol_for_id[inst_id] = meta.symbol_full

    def _fetch_rates(
        self,
        universe: TrackedUniverse,
        *,
        prior: dict[int, LiveRate],
    ) -> dict[int, LiveRate]:
        """Fetch live rates; on failure, fall back to the previous cycle's cache.

        eToro occasionally returns 5xx for whole-batch rate calls; rather
        than blanking an entire cycle (which would silence every signal),
        we serve the previous successful rates and tag the cycle as
        degraded.
        """
        if not universe.instrument_ids:
            return {}
        try:
            rates = fetch_rates(self._etoro, universe.instrument_ids)
        except Exception as exc:  # noqa: BLE001
            cached = {
                inst_id: prior[inst_id]
                for inst_id in universe.instrument_ids
                if inst_id in prior
            }
            if cached:
                self._log.warning(
                    "[market] rates fetch failed (%s) — using %d cached rate(s)",
                    exc, len(cached),
                )
                return cached
            self._log.error("[market] rates fetch failed: %s (no cache)", exc)
            return {}
        self._log.info("[market] fetched %d/%d rate(s)", len(rates), len(universe.instrument_ids))
        return rates

    def _fetch_candles(self, universe: TrackedUniverse) -> dict[int, list[Candle]]:
        out: dict[int, list[Candle]] = {}
        for inst_id in universe.instrument_ids:
            try:
                out[inst_id] = fetch_candles(
                    self._etoro,
                    inst_id,
                    interval=self._cfg.operations.candle_interval,
                    count=self._cfg.operations.candle_count,
                )
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "[market] candles failed for %s: %s",
                    universe.symbol_for_id.get(inst_id, inst_id), exc,
                )
        return out

    def _fetch_portfolio(self) -> PortfolioSnapshot:
        try:
            return fetch_portfolio(self._etoro, self._cfg.env_segment)
        except Exception as exc:  # noqa: BLE001
            self._log.error("[portfolio] PnL fetch failed: %s", exc)
            return PortfolioSnapshot(credit=0.0, unrealized_pnl=0.0)

    def _log_portfolio_summary(self, summary: dict[str, float], bot_owned_count: int) -> None:
        self._log.info(
            "[portfolio] equity=$%.2f  available=$%.2f  invested=$%.2f  pnl=$%.2f  bot_owned=%d",
            summary.get("equity", 0.0),
            summary.get("available_cash", 0.0),
            summary.get("total_invested", 0.0),
            summary.get("profit_loss", 0.0),
            bot_owned_count,
        )

    def _log_results(self, results: Iterable[ExecutionResult]) -> None:
        results = list(results)
        if not results:
            self._log.info("[exec] nothing to do")
            return
        for r in results:
            self._log.info(
                "[exec] %-10s %-6s %-9s — %s",
                r.status.upper(), r.action, r.request_symbol, r.detail,
            )

    # ------------------------------------------------------------------
    # Autonomous-tuner integration
    # ------------------------------------------------------------------

    def _raw_scores_for_universe(
        self,
        candles_by_instrument: dict[int, list[Candle]],
    ) -> list[float]:
        """Compute the raw_score across EVERY tracked symbol this cycle.

        The signals layer only returns symbols *above* the threshold;
        the autotuner needs the *full* distribution (including symbols
        well below the bar) so the LLM can detect a mis-calibrated
        gate (e.g. everyone clustered at 0.30 when the threshold is 0.40).
        """
        scores: list[float] = []
        for candles in candles_by_instrument.values():
            if not candles or len(candles) < 5:
                continue
            closes = [c.close for c in candles if c.close > 0]
            highs = [c.high if c.high > 0 else c.close for c in candles]
            lows = [c.low if c.low > 0 else c.close for c in candles]
            if not closes:
                continue
            try:
                result = evaluate_ensemble(
                    closes=closes, highs=highs, lows=lows,
                    cfg=self._cfg.strategy,
                )
            except Exception:  # noqa: BLE001 - never let a single symbol kill the cycle
                continue
            scores.append(float(result.raw_score))
        return scores

    def _run_position_review(
        self,
        bot_owned_positions: list,
        symbol_for_id: dict,
        rate_cache: dict,
    ) -> list:
        """Evaluate every open bot position against the trigger thresholds.

        Returns the list of :class:`PositionReview` for positions that
        fired ≥1 trigger. The list is passed to the LLM as the
        ``performance.position_reviews`` block so triggered positions
        force a decision instead of being silently held.
        """
        if self._position_reviewer is None or not bot_owned_positions:
            return []
        open_states = (
            self._performance.open_states_by_position()
            if self._performance else {}
        )
        return self._position_reviewer.evaluate(
            bot_owned_positions=bot_owned_positions,
            symbol_for_id=symbol_for_id,
            perf_open_states=open_states,
            live_rates=rate_cache,
        )

    def _directives(self) -> Directives:
        """Return the live directives snapshot (or an all-default one)."""
        if self._directives_store is None:
            return Directives()
        return self._directives_store.current()

    def _build_by_symbol_projection(
        self,
        candidates: list,
        bot_owned_positions: list,
        ctx: "CycleContext",
    ) -> dict[str, dict]:
        """Build the per-symbol track-record projection for the LLM payload.

        We only include symbols the LLM is likely to act on this cycle
        (candidates + currently-owned). Closed-trade aggregates older
        than that are noise.
        """
        if self._performance is None:
            return {}
        wanted: list[str] = []
        for c in candidates:
            sym = getattr(c, "symbol", None)
            if sym:
                wanted.append(str(sym).upper())
        for p in bot_owned_positions:
            sym = ctx.universe.symbol_for_id.get(p.instrument_id)
            if sym:
                wanted.append(sym.upper())
        if not wanted:
            return {}
        return project_by_symbol_history(
            by_symbol=self._performance.by_symbol(),
            symbols_of_interest=wanted,
        )

    def _build_autotune_evidence(
        self,
        snapshot: PortfolioSnapshot,
    ) -> dict | None:
        """Materialise the per-cycle evidence digest for the LLM. None if disabled."""
        if self._autotune is None:
            return None
        # Use the live cfg-aware helper now that the bound method exists.
        # (Replaces the placeholder static helper above for the actual
        # data path; the static helper is kept for tests.)
        recent_pnl = self._recent_realized_pnl()
        open_pnl = float(snapshot.unrealized_pnl or 0.0)
        evidence = self._autotune.build_evidence(
            cfg=self._cfg,
            recent_realized_pnl=recent_pnl,
            open_position_pnl_total=open_pnl,
        )
        return evidence.to_dict()

    def _recent_realized_pnl(self, *, limit: int = 10) -> list[dict]:
        """Recent CLOSE-action trade-history entries with extracted P&L hints.

        The history log doesn't carry realised-P&L numerically yet —
        the executor records ``detail`` and ``status`` only. For the
        LLM's purpose we project the last ``limit`` close-style entries
        (CLOSE / panic_close) and their statuses so the model can
        reason about trade frequency and failure rate.
        """
        if self._history is None:
            return []
        try:
            entries = self._history.tail(limit=int(limit))
        except Exception:  # noqa: BLE001
            return []
        out: list[dict] = []
        for e in entries:
            d = e.to_dict() if hasattr(e, "to_dict") else dict(e)
            out.append({
                "timestamp": d.get("timestamp"),
                "action": d.get("action"),
                "status": d.get("status"),
                "symbol": d.get("symbol"),
                "amount_usd": d.get("amount_usd"),
                "detail": d.get("detail"),
            })
        return out

    def _apply_autotune(self, decision: DecisionResult) -> None:
        """Apply any tuning request the LLM attached + emit the alert."""
        if self._autotune is None:
            return
        tuning = getattr(decision, "tuning", None)
        if tuning is None or tuning.is_empty:
            return
        applied = self._autotune.apply(tuning, cfg=self._cfg)
        if not applied:
            return
        diff = render_tune_diff(applied)
        rationale_lines = [a.rationale for a in applied if a.rationale]
        body = diff
        if tuning.reason:
            body += f"\n\nReason: {tuning.reason}"
        if rationale_lines:
            body += "\n\nPer-change:\n" + "\n".join(f"• {r}" for r in rationale_lines)
        self._log.info("[autotune] %s", diff)
        self._emit(
            AlertType.STRATEGY_AUTOTUNED,
            title=f"AUTOTUNED ({len(applied)} change{'s' if len(applied) != 1 else ''})",
            body=body,
        )

    # ------------------------------------------------------------------
    # Telemetry & trade-history publishing (used by the Telegram service)
    # ------------------------------------------------------------------

    def _publish_universe(self, universe: TrackedUniverse) -> None:
        if self._telemetry is None:
            return
        ids = list(universe.instrument_ids)
        symbols = [universe.symbol_for_id.get(i, str(i)) for i in ids]
        reasons = {
            universe.symbol_for_id.get(i, str(i)): universe.reason_for_id.get(i, "")
            for i in ids
        }
        self._telemetry.update_universe(
            instrument_ids=ids,
            symbols=symbols,
            base_count=universe.base_count,
            llm_count=universe.llm_count,
            reasons=reasons,
            source_counts=universe.source_counts,
            rejected=universe.rejected,
        )

    def _publish_portfolio(
        self,
        snapshot: PortfolioSnapshot,
        *,
        bot_owned_position_ids: list[int],
        symbol_for_id: dict[int, str],
    ) -> None:
        if self._telemetry is None:
            return
        positions_view = []
        for p in snapshot.positions:
            positions_view.append({
                "position_id": p.position_id,
                "instrument_id": p.instrument_id,
                "symbol": symbol_for_id.get(p.instrument_id, str(p.instrument_id)),
                "is_buy": p.is_buy,
                "open_rate": p.open_rate,
                "amount": p.amount,
                "leverage": p.leverage,
                "pnl": p.pnl,
                "is_bot_owned": p.position_id in bot_owned_position_ids,
                "is_mirror": p.is_mirror,
            })
        self._telemetry.update_portfolio(
            summary=compute_account_summary(snapshot),
            positions=positions_view,
            bot_owned_position_ids=bot_owned_position_ids,
        )

    def _publish_decision(self, decision: DecisionResult) -> None:
        if self._telemetry is None:
            return
        actions = [
            {
                "action": r.action,
                "symbol": r.symbol,
                "instrument_id": r.instrument_id,
                "amount_usd": r.amount_usd,
                "position_id": r.position_id,
            }
            for r in decision.requests
        ]
        self._telemetry.update_decision(
            summary=decision.summary,
            llm_used=decision.llm_used,
            actions=actions,
        )

    def _record_history(self, results: Iterable[ExecutionResult]) -> None:
        if self._history is None:
            return
        ts = utc_now_iso()
        results = list(results)
        for r in results:
            if r.status == "skipped":
                # Skipped trades are noisy; the operator only cares about
                # actual eToro round-trips.
                continue
            self._history.append(TradeHistoryEntry(
                timestamp=ts,
                action=r.action,
                status=r.status,
                symbol=r.request_symbol,
                instrument_id=r.instrument_id,
                amount_usd=r.amount_usd,
                order_id=r.order_id,
                position_id=r.position_id,
                detail=r.detail,
            ))
        self._emit_trade_alerts(results)

    # ------------------------------------------------------------------
    # Alert emission (no-ops when ``self._alerts is None``)
    # ------------------------------------------------------------------

    def _emit(self, alert_type: AlertType, *, title: str, body: str = "") -> None:
        if self._alerts is None:
            return
        try:
            self._alerts.emit(alert_type, title=title, body=body)
        except Exception as exc:  # noqa: BLE001 - alerts must never block a cycle
            self._log.warning("[alerts] emit failed for %s: %s", alert_type.value, exc)

    # ------------------------------------------------------------------
    # Order-lifecycle plumbing
    # ------------------------------------------------------------------

    def _track_placed_orders(
        self, results: list[ExecutionResult], ctx: CycleContext,
    ) -> None:
        """Register successfully-placed orders with the position monitor.

        Both BUY (market-open) and CLOSE (market-close) orders are
        tracked so the stuck-order pipeline can later detect either
        kind sitting unfilled past its session-grace window.
        """
        now_utc = datetime.now(timezone.utc)
        now_mono = time.monotonic()
        for r in results:
            if r.status != "ok" or r.order_id is None or r.instrument_id is None:
                continue
            asset_class = self._asset_class_for(r.instrument_id, r.request_symbol, ctx)
            if r.action == "BUY":
                self._monitor.track_open(
                    order_id=r.order_id,
                    instrument_id=r.instrument_id,
                    symbol=r.request_symbol,
                    asset_class=asset_class,
                    amount_usd=r.amount_usd or 0.0,
                    placed_at_utc=now_utc,
                    placed_at_monotonic=now_mono,
                )
            elif r.action == "CLOSE" and r.position_id is not None:
                self._monitor.track_close(
                    order_id=r.order_id,
                    position_id=r.position_id,
                    instrument_id=r.instrument_id,
                    symbol=r.request_symbol,
                    asset_class=asset_class,
                    placed_at_utc=now_utc,
                    placed_at_monotonic=now_mono,
                )

    def _asset_class_for(
        self, instrument_id: int, symbol: str, ctx: CycleContext,
    ) -> AssetClass:
        meta = ctx.instrument_metas.get(instrument_id)
        return asset_class_for(meta, symbol=symbol)

    # ------------------------------------------------------------------
    # Performance tracking hooks (no-ops when tracker not configured)
    # ------------------------------------------------------------------

    def _record_performance_opens(
        self, adopted: list, ctx: CycleContext,
    ) -> None:
        if self._performance is None or not adopted:
            return
        for pos in adopted:
            symbol = ctx.universe.symbol_for_id.get(pos.instrument_id, str(pos.instrument_id))
            asset_class = self._asset_class_for(pos.instrument_id, symbol, ctx)
            self._performance.record_open(
                position_id=int(pos.position_id),
                instrument_id=int(pos.instrument_id),
                symbol=symbol,
                asset_class=asset_class.value,
                is_buy=bool(pos.is_buy),
                amount_usd=float(pos.amount),
                units=float(getattr(pos, "units", 0.0) or 0.0),
                open_rate=float(pos.open_rate),
            )

    def _record_performance_vanished(self, snapshot: PortfolioSnapshot) -> None:
        """Detect bot-owned positions that vanished from the broker
        (SL/TP triggered, user manually closed, etc.) and book them
        as realized trades. Also prunes ``state.bot_owned_positions``
        so the set doesn't grow unbounded with stale IDs."""
        if not self._state.bot_owned_positions:
            return
        snapshot_ids = {p.position_id for p in snapshot.positions}
        vanished = [
            pid for pid in list(self._state.bot_owned_positions)
            if pid not in snapshot_ids
        ]
        if not vanished:
            return
        for pid in vanished:
            if self._performance is not None:
                trade = self._performance.record_close(
                    position_id=pid, reason="external",
                )
                if trade is not None:
                    self._log.info(
                        "[perf] booked vanished position %s P/L=$%.2f",
                        trade.symbol, trade.realized_pnl_usd,
                    )
            self._state.remove_owned(pid)

    def _record_performance_observe(
        self,
        bot_owned_positions: list,
        *,
        summary: dict,
    ) -> None:
        if self._performance is None:
            return
        self._performance.observe_positions(
            bot_positions=bot_owned_positions,
            equity=summary.get("equity"),
            account_unrealized_pnl=summary.get("profit_loss"),
        )

    def _cancel_stuck_orders(self, snapshot: PortfolioSnapshot) -> None:
        """Detect stuck pending orders and try to cancel them.

        Telegram alerts (``ORDER_STUCK_CANT_CANCEL``) fire ONLY when
        the cancel itself is refused AND the post-failure status
        recheck confirms the order is still non-terminal. Successful
        auto-cancels and "raced and lost — actually filled" outcomes
        are logged but not alerted, per the project's noise budget.
        """
        if self._monitor.tracked_count == 0:
            return
        grace = int(self._cfg.operations.pending_grace_seconds_after_open)
        now_utc = datetime.now(timezone.utc)
        stuck = self._monitor.find_stuck(
            snapshot,
            now_utc=now_utc,
            grace_seconds_after_open=grace,
        )
        if not stuck:
            return
        self._log.warning(
            "[monitor] %d stuck order(s) detected: %s",
            len(stuck),
            ", ".join(
                f"{s.tracked.action} {s.tracked.symbol}#{s.tracked.order_id} "
                f"(waited {s.waited_seconds}s, in-session {s.in_session_seconds}s)"
                for s in stuck
            ),
        )
        if not self._cfg.operations.cancel_stuck_orders_enabled:
            self._log.info(
                "[monitor] cancel_stuck_orders_enabled=false — not cancelling",
            )
            return
        for s in stuck:
            result = self._monitor.cancel(
                self._etoro, env=self._cfg.env_segment, stuck=s,
            )
            if result.alert:
                self._emit_stuck_alert(s, result)

    def _emit_stuck_alert(self, stuck: StuckOrder, result: CancelResult) -> None:
        tracked = result.tracked
        status_name = (
            result.final_status.name if result.final_status is not None else "UNKNOWN"
        )
        body = (
            f"orderID={tracked.order_id}  •  {tracked.action} {tracked.symbol}  •  "
            f"waited {stuck.waited_seconds}s (in-session {stuck.in_session_seconds}s)\n"
            f"final status: {status_name}\n"
            f"reason: {result.detail}"
        )
        self._emit(
            AlertType.ORDER_STUCK_CANT_CANCEL,
            title=f"STUCK ORDER — {tracked.symbol} ({tracked.action})",
            body=body,
        )

    def _emit_trade_alerts(self, results: list[ExecutionResult]) -> None:
        for r in results:
            if r.status == "skipped":
                continue
            sym = r.request_symbol or "?"
            amt = f"${r.amount_usd:.2f}" if r.amount_usd is not None else "—"
            if r.status == "ok" and r.action == "BUY":
                self._emit(
                    AlertType.TRADE_OPENED,
                    title=f"OPENED {sym}",
                    body=f"{amt} long  •  orderID={r.order_id}",
                )
            elif r.status == "ok" and r.action == "CLOSE":
                self._emit(
                    AlertType.TRADE_CLOSED,
                    title=f"CLOSED {sym}",
                    body=f"positionID={r.position_id}  •  orderID={r.order_id}",
                )
            elif r.status in {"failed", "ambiguous", "rate_limited"}:
                self._emit(
                    AlertType.TRADE_FAILED,
                    title=f"{r.action} {sym} — {r.status.upper()}",
                    body=r.detail or "(no detail)",
                )

    def emit_kill_switch_if_changed(self, *, halted: bool) -> None:
        """Called by the loop after risk eval; emits on False→True edge."""
        if halted and not self._last_halted:
            self._emit(
                AlertType.DAILY_LOSS_HALT,
                title="DAILY LOSS KILL SWITCH",
                body=(
                    f"daily_loss_stop_usd reached "
                    f"(${self._cfg.guardrails.daily_loss_stop_usd:.0f}). "
                    "New BUYs blocked until next UTC day."
                ),
            )
        self._last_halted = bool(halted)

    def emit_ai_state_if_changed(self, *, decision_used_llm: bool) -> None:
        """Edge-detect Azure availability: only alert on transitions."""
        # We treat "decision used the LLM" as a proxy for "Azure is up". A
        # cycle that legitimately had no candidates also reports llm_used
        # = False, so we skip when there were no requests by checking the
        # ai_client presence first.
        if self._ai is None:
            return
        currently_available = bool(decision_used_llm)
        if self._last_ai_available is None:
            self._last_ai_available = currently_available
            return
        if currently_available != self._last_ai_available:
            if currently_available:
                self._emit(
                    AlertType.AI_UNAVAILABLE,
                    title="AI RECOVERED",
                    body="Azure / LLM responding again.",
                )
            else:
                self._emit(
                    AlertType.AI_UNAVAILABLE,
                    title="AI UNAVAILABLE",
                    body=(
                        "Azure / LLM call failed. Cycle fell back to "
                        "deterministic signals (or vetoed BUYs, depending "
                        "on config.veto_on_unavailable)."
                    ),
                )
            self._last_ai_available = currently_available

    def emit_universe_change(self, universe: TrackedUniverse) -> None:
        """Diff against the last published universe; emit on add/remove.

        Also emits the opt-in :data:`AlertType.UNIVERSE_REJECTED` when
        the refresh produced activity-filter rejections — operators
        tuning ``[universe] min_atr_pct`` / ``max_spread_pct`` rely on
        this to see *why* a promising news ticker didn't make the cut.
        """
        current_symbols = tuple(
            sorted(universe.symbol_for_id.get(i, str(i)) for i in universe.instrument_ids)
        )
        self._emit_universe_rejected_if_any(universe)
        if not self._last_universe_symbols:
            self._last_universe_symbols = current_symbols
            return
        if current_symbols == self._last_universe_symbols:
            return
        added = sorted(set(current_symbols) - set(self._last_universe_symbols))
        removed = sorted(set(self._last_universe_symbols) - set(current_symbols))
        bits = []
        if added:
            bits.append("added: " + ", ".join(added))
        if removed:
            bits.append("removed: " + ", ".join(removed))
        self._emit(
            AlertType.UNIVERSE_CHANGED,
            title=f"UNIVERSE CHANGED ({len(current_symbols)} now tracked)",
            body=" • ".join(bits) if bits else "rotation refreshed",
        )
        self._last_universe_symbols = current_symbols

    def _emit_universe_rejected_if_any(self, universe: TrackedUniverse) -> None:
        """Surface up to the first 10 rejection reasons. No-op if none.

        Bounded to avoid spamming a chat when 50 candidates fail the
        filter on a quiet market day; ``/universe`` shows the full
        list for an operator that wants details.
        """
        rejected = getattr(universe, "rejected", None) or {}
        if not rejected:
            return
        sample = list(rejected.items())[:10]
        body = "\n".join(f"• {sym}: {reason}" for sym, reason in sample)
        if len(rejected) > 10:
            body += f"\n… and {len(rejected) - 10} more (see /universe)"
        self._emit(
            AlertType.UNIVERSE_REJECTED,
            title=f"UNIVERSE REJECTED ({len(rejected)})",
            body=body,
        )

    def emit_cycle_error(self, message: str) -> None:
        self._emit(
            AlertType.CYCLE_ERROR,
            title="CYCLE CRASHED",
            body=message[:1000] if message else "(no detail)",
        )
