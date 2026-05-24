"""One full trading cycle: fetch → decide → risk-gate → execute → monitor.

Pulled out of ``src.main`` so the entry-point file stays a thin wrapper
and the loop logic is independently testable. The cycle is deliberately
linear; readability beats cleverness when something goes wrong at 3 AM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .ai.azure_client import AzureFoundryClient
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
from .execution.executor import ExecutionResult, TradeExecutor
from .execution.monitor import PositionMonitor
from .state import BotState
from .strategy.decisions import DecisionEngine, DecisionResult, render_decisions
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
        # State-change tracking so we only emit AI_UNAVAILABLE / DAILY_LOSS_HALT
        # alerts on edges, not every cycle.
        self._last_ai_available: bool | None = None
        self._last_halted: bool = False
        self._last_universe_symbols: tuple[str, ...] = ()

    def initial_universe(self) -> CycleContext:
        universe = self._universe_builder.build()
        ctx = CycleContext(
            universe=universe,
            last_universe_refresh_monotonic=time.monotonic(),
        )
        self._log.info(
            "[universe] tracking %d instrument(s): base=%d, llm=%d",
            len(universe), universe.base_count, universe.llm_count,
        )
        self._cache_instrument_metadata(universe, ctx.instrument_metas)
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
        self._maybe_refresh_universe(ctx)

        rates = self._fetch_rates(ctx.universe, prior=ctx.rate_cache)
        ctx.rate_cache.update(rates)
        snapshot = self._fetch_portfolio()
        self._monitor.reconcile(snapshot, self._state)
        bot_owned_positions = [
            p for p in snapshot.positions
            if p.position_id in self._state.bot_owned_positions and p.mirror_id == 0
        ]
        self._publish_portfolio(snapshot, bot_owned_position_ids=[
            p.position_id for p in bot_owned_positions
        ], symbol_for_id=ctx.universe.symbol_for_id)

        candles_by_instrument = self._fetch_candles(ctx.universe)
        candidates = build_candidates(
            cfg=self._cfg.strategy,
            candles_by_instrument=candles_by_instrument,
            symbol_for_id=ctx.universe.symbol_for_id,
            bot_owned_instrument_ids={p.instrument_id for p in bot_owned_positions},
        )
        self._log.info(
            "[signals] %d candidate(s): %s",
            len(candidates),
            ", ".join(f"{c.action} {c.symbol}({c.strength:.2f})" for c in candidates) or "—",
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

        summary = compute_account_summary(snapshot)
        rules_payload = build_rules_payload(
            strategy=self._cfg.strategy,
            guardrails=self._cfg.guardrails,
            tools=[],
        )
        decision = self._decision_engine.decide(
            candidates=candidates,
            portfolio_summary=summary,
            bot_owned_positions=bot_owned_positions,
            symbol_for_id=ctx.universe.symbol_for_id,
            tool_results=tool_results,
            cross_asset_regime=cross_asset_regime,
            strategy_rules=rules_payload,
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

        verdicts = self._risk.evaluate(
            requests=decision.requests,
            state=self._state,
            current_equity=summary.get("equity"),
            bot_owned_position_count=len(bot_owned_positions),
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

        results = self._executor.execute_all(
            verdicts=verdicts, rates=rates, state=self._state,
        )
        for r in results:
            if (
                r.status == "ok"
                and r.action == "BUY"
                and r.order_id is not None
                and r.instrument_id is not None
            ):
                self._monitor.track_open(
                    order_id=r.order_id,
                    instrument_id=r.instrument_id,
                    symbol=r.request_symbol,
                    amount_usd=r.amount_usd or 0.0,
                    placed_at_monotonic=time.monotonic(),
                )
        self._record_history(results)

        self._log_portfolio_summary(summary, len(bot_owned_positions))
        self._log_results(results)
        if self._telemetry is not None:
            self._telemetry.mark_cycle_finished()

    # ------------------------------------------------------------------

    def _maybe_refresh_universe(self, ctx: CycleContext) -> None:
        elapsed_min = (time.monotonic() - ctx.last_universe_refresh_monotonic) / 60.0
        if elapsed_min < self._cfg.operations.universe_refresh_minutes:
            return
        self._log.info("[universe] refreshing (after %.1f min)", elapsed_min)
        ctx.universe = self._universe_builder.build()
        ctx.last_universe_refresh_monotonic = time.monotonic()
        self._log.info(
            "[universe] now tracking %d instrument(s): base=%d, llm=%d",
            len(ctx.universe), ctx.universe.base_count, ctx.universe.llm_count,
        )
        self._cache_instrument_metadata(ctx.universe, ctx.instrument_metas)
        self._publish_universe(ctx.universe)
        self.emit_universe_change(ctx.universe)

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
    # Telemetry & trade-history publishing (used by the Telegram service)
    # ------------------------------------------------------------------

    def _publish_universe(self, universe: TrackedUniverse) -> None:
        if self._telemetry is None:
            return
        ids = list(universe.instrument_ids)
        symbols = [universe.symbol_for_id.get(i, str(i)) for i in ids]
        self._telemetry.update_universe(
            instrument_ids=ids,
            symbols=symbols,
            base_count=universe.base_count,
            llm_count=universe.llm_count,
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
        """Diff against the last published universe; emit on add/remove."""
        current_symbols = tuple(
            sorted(universe.symbol_for_id.get(i, str(i)) for i in universe.instrument_ids)
        )
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

    def emit_cycle_error(self, message: str) -> None:
        self._emit(
            AlertType.CYCLE_ERROR,
            title="CYCLE CRASHED",
            body=message[:1000] if message else "(no detail)",
        )
