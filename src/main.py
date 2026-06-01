"""Bot entry point — wires components, owns the outer loop and signals.

Run with::

    python -m src.main

The cycle logic itself lives in :mod:`src.cycle`. This file's job is
purely orchestration: load config, build clients, install signal
handlers, sleep between cycles, shut down cleanly. It also boots the
internal HTTP control server (the surface the Telegram service talks
to) and restores persisted bot state if a previous process saved it.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from .ai.azure_client import AzureFoundryClient, AzureUnavailable
from .ai.usage_tracker import LLMUsageTracker
from .alerts import AlertHub, AlertSubscriptions, safety_only_default
from .config import AppConfig, DEFAULT_CONFIG_DB_PATH, PROJECT_ROOT, load_config, summarize_config
from .config_store import ConfigStore, open_store
from .control.controller import BotController
from .control.server import ControlHTTPServer
from .cycle import CycleRunner
from .etoro.client import EtoroClient
from .etoro.identity import fetch_identity
from .etoro.instrument_cache import InstrumentCache
from .execution.dynamic_stops import DynamicStopsStore
from .execution.executor import TradeExecutor
from .execution.monitor import PositionMonitor
from .fundamentals import FundamentalsCache, build_fundamentals_cache
from .logging_setup import configure_logging, get_logger
from .news.candidate_store import CandidateStore
from .news.factory import build_news_pipeline
from .news.scheduler import NewsScheduler
from .performance import PerformanceTracker
from .persistence import StatePersistence
from .state import BotState
from .strategy.activity_filter import ActivityFilter
from .strategy.autotune import AutotuneState
from .strategy.decisions import DecisionEngine
from .strategy.directives import Directives, DirectivesStore
from .strategy.position_review import PositionReviewer, PositionReviewConfig
from .strategy.risk import RiskEvaluator
from .strategy.tool_orchestration import ToolOrchestrator
from .strategy.universe import UniverseBuilder
from .telemetry import TelemetryStore
from .trade_history import TradeHistoryLog


class TradingBot:
    def __init__(
        self,
        cfg: AppConfig,
        project_root: Path = PROJECT_ROOT,
        *,
        config_store: ConfigStore | None = None,
    ) -> None:
        self.cfg = cfg
        self._project_root = project_root
        configure_logging(cfg.logging, project_root)
        self.log = get_logger("main", tag="main")
        self._stop = False
        # Threading event mirrors `self._stop` so subsystems running in
        # blocking code (cycle runner, universe probe, news scan) can
        # poll a single source of truth and bail out early instead of
        # finishing every step after a Ctrl-C.
        self._stop_event = threading.Event()
        self._signal_count = 0

        self._config_store = config_store
        self.state = BotState()
        self._persistence = StatePersistence(
            project_root / "data" / "bot_state.json",
            logger=get_logger("persistence", tag="state"),
        )
        self._restore_persisted_state()

        self.cache = InstrumentCache.load(project_root / "data" / "instrument_cache.json")
        self._etoro = EtoroClient(
            credentials=cfg.etoro,
            request_timeout_seconds=cfg.operations.request_timeout_seconds,
            logger=get_logger("etoro.client", tag="etoro"),
        )
        self._ai_client: AzureFoundryClient | None = self._build_ai_client()
        self._usage_tracker = self._build_usage_tracker()
        if self._ai_client is not None and self._usage_tracker is not None:
            self._ai_client.set_usage_tracker(self._usage_tracker)
        self.telemetry = TelemetryStore()
        self.history = TradeHistoryLog(
            project_root / "data" / "trade_history.jsonl",
            logger=get_logger("trade_history", tag="history"),
        )
        self.alerts = self._build_alert_hub()
        self.controller = BotController(
            cfg=cfg,
            state=self.state,
            etoro=self._etoro,
            ai_client=self._ai_client,
            telemetry=self.telemetry,
            history=self.history,
            persistence=self._persistence,
            config_store=self._config_store,
            alerts=self.alerts,
            logger=get_logger("control", tag="control"),
        )
        if self._initial_paused:
            self.controller.set_paused_initial(True)
        self._tool_orchestrator = self._build_tool_orchestrator()
        if self._tool_orchestrator is not None:
            self.controller.set_tool_orchestrator(self._tool_orchestrator)
        self._news_store, self._news_scheduler = self._build_news_pipeline()
        if self._news_store is not None:
            self.controller.set_news_store(self._news_store, self._news_scheduler)
        self._fundamentals = self._build_fundamentals_cache()
        if self._fundamentals is not None:
            self.controller.set_fundamentals(self._fundamentals)
        self._earnings_calendar = self._build_earnings_calendar()
        self._autotune = self._build_autotune_state()
        self.controller.set_autotune_state(self._autotune)
        self._performance = PerformanceTracker(
            project_root / "data",
            logger=get_logger("performance", tag="perf"),
        )
        self.controller.set_performance_tracker(self._performance)
        pr_dc = getattr(cfg, "position_review", None)
        self._position_review_cfg = PositionReviewConfig(
            drawdown_pct=float(pr_dc.drawdown_pct) if pr_dc else 2.0,
            pullback_pct=float(pr_dc.pullback_pct) if pr_dc else 3.0,
            stale_hold_minutes=float(pr_dc.stale_hold_minutes) if pr_dc else 60.0,
            stale_threshold_pct=float(pr_dc.stale_threshold_pct) if pr_dc else 0.5,
            max_hold_minutes=float(pr_dc.max_hold_minutes) if pr_dc else 240.0,
        )
        self._directives_store = DirectivesStore(
            logger=get_logger("strategy.directives", tag="directives"),
        )
        persisted_directives = self._persistence.load_directives()
        if persisted_directives:
            self._directives_store.restore(persisted_directives)
            self.log.info(
                "[directives] restored: no_overnight=%s hold_ceiling=%dm "
                "blocked_symbols=%d blocked_sectors=%d total_cap=$%.0f notes=%d chars",
                self._directives_store.current().no_overnight,
                self._directives_store.current().hold_ceiling_minutes,
                len(self._directives_store.current().blocked_symbols),
                len(self._directives_store.current().blocked_sectors),
                self._directives_store.current().max_total_account_invested_usd,
                len(self._directives_store.current().notes),
            )
        self.controller.set_directives_store(self._directives_store)
        if self._usage_tracker is not None:
            self.controller.set_token_usage_tracker(self._usage_tracker)
        self._control_server = self._maybe_start_control_server()
        self._runner = self._build_runner()

    # ------------------------------------------------------------------

    def stop(self, *_: object) -> None:
        """Signal handler. First press → graceful; second → hard exit.

        With Phase 2 a single cycle includes several synchronous HTTP
        calls (universe probe + LLM decision) that can run for tens of
        seconds. Operators expect Ctrl-C to mean "I'm done waiting" the
        first time, and "no, really, exit now" the second time.
        """
        self._signal_count += 1
        if self._signal_count == 1:
            self.log.info(
                "shutdown signal received — stopping after current step "
                "(press Ctrl-C again to force exit)",
            )
            self._stop = True
            self._stop_event.set()
            return
        # Second signal: restore the default handler so a *third* Ctrl-C
        # gives the operator their normal escape hatch, then trigger an
        # immediate exit. We can't raise KeyboardInterrupt cleanly from
        # here because we're already inside a signal frame; os._exit
        # skips finalizers but that's the right call when the user has
        # asked twice.
        self.log.warning(
            "second shutdown signal — forcing immediate exit (130)",
        )
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except (ValueError, OSError):
            # Restoring defaults is best-effort; we're exiting anyway.
            pass
        os._exit(130)

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        self.log.info("etrader starting — %s", summarize_config(self.cfg))
        if self.controller.paused:
            self.log.warning(
                "[main] starting in PAUSED state (restored from %s) — "
                "send /resume from Telegram or POST /resume to wake up",
                self._persistence.path,
            )
        self._announce_identity()

        ctx = self._runner.initial_universe()

        try:
            while not self._stop and not self.controller.stop_requested:
                if self.controller.paused:
                    self._interruptible_sleep(1.0)
                    continue
                self.state.cycle_count += 1
                cycle_start = time.monotonic()
                try:
                    with self.controller.lock:
                        self._runner.run_one(ctx)
                    self.controller.persist_state()
                except Exception:  # noqa: BLE001 — never let one cycle kill the bot
                    msg = traceback.format_exc()
                    self.log.error("cycle %d crashed:\n%s", self.state.cycle_count, msg)
                    last_line = msg.splitlines()[-1] if msg else "unknown"
                    self.telemetry.mark_cycle_error(last_line)
                    self._runner.emit_cycle_error(last_line)
                elapsed = time.monotonic() - cycle_start
                sleep_for = max(1.0, self.cfg.operations.check_interval_seconds - elapsed)
                if not self._stop and not self.controller.stop_requested:
                    self.log.debug("sleeping %.1fs before next cycle", sleep_for)
                    self._interruptible_sleep(sleep_for)
        finally:
            self._shutdown()

        self.log.info("etrader stopped after %d cycle(s)", self.state.cycle_count)
        return 0

    # ------------------------------------------------------------------

    def _build_runner(self) -> CycleRunner:
        risk = RiskEvaluator(
            self.cfg.guardrails,
            directives_provider=self._directives_store.current,
        )
        dynamic_stops = DynamicStopsStore(
            default_stop_loss_pct=self.cfg.guardrails.default_stop_loss_pct,
            default_take_profit_pct=self.cfg.guardrails.default_take_profit_pct,
            logger=get_logger("execution.dynamic_stops", tag="stops"),
        )
        restored_stops = self._persistence.load_dynamic_stops()
        if restored_stops:
            dynamic_stops.restore(restored_stops)
        self.controller.set_dynamic_stops(dynamic_stops)
        executor = TradeExecutor(
            client=self._etoro,
            env=self.cfg.env_segment,
            guardrails=self.cfg.guardrails,
            operations=self.cfg.operations,
            logger=get_logger("execution.executor", tag="exec"),
            dynamic_stops=dynamic_stops,
        )
        monitor = PositionMonitor(
            logger=get_logger("execution.monitor", tag="monitor"),
            dynamic_stops=dynamic_stops,
        )
        position_reviewer = PositionReviewer(
            self._position_review_cfg,
            logger=get_logger("strategy.position_review", tag="review"),
        )
        decision_engine = DecisionEngine(
            ai_cfg=self.cfg.ai,
            guardrails=self.cfg.guardrails,
            ai_client=self._ai_client,
            logger=get_logger("strategy.decisions", tag="ai"),
        )
        activity_filter = ActivityFilter(self.cfg.universe)
        universe_builder = UniverseBuilder(
            self.cfg.universe,
            self.cfg.operations,
            cache=self.cache,
            candidate_store=self._news_store,
            activity_filter=activity_filter,
            ai_client=self._ai_client,
            etoro_client=self._etoro,
            logger=get_logger("strategy.universe", tag="universe"),
            is_stopping=self._stop_event.is_set,
        )
        return CycleRunner(
            self.cfg,
            etoro=self._etoro,
            ai_client=self._ai_client,
            state=self.state,
            universe_builder=universe_builder,
            risk=risk,
            executor=executor,
            monitor=monitor,
            decision_engine=decision_engine,
            telemetry=self.telemetry,
            history=self.history,
            tool_orchestrator=self._tool_orchestrator,
            alerts=self.alerts,
            news_scheduler=self._news_scheduler,
            fundamentals_cache=self._fundamentals,
            earnings_calendar=self._earnings_calendar,
            autotune_state=self._autotune,
            performance=self._performance,
            dynamic_stops=dynamic_stops,
            position_reviewer=position_reviewer,
            directives_store=self._directives_store,
            stop_event=self._stop_event,
            log=get_logger("cycle", tag="cycle"),
        )

    def _build_autotune_state(self) -> AutotuneState:
        """Construct the autonomous-tuner state machine.

        Restores any persisted drought counters + tuning log from
        ``bot_state.json`` so a restart doesn't lose the LLM's
        accumulated context.
        """
        autotune = AutotuneState(
            config_store=self._config_store,
            logger=get_logger("strategy.autotune", tag="autotune"),
        )
        persisted = self._persistence.load_autotune()
        if persisted:
            autotune.restore(persisted)
            self.log.info(
                "[autotune] restored: drought_candidates=%d drought_trades=%d "
                "previous_tunings=%d",
                persisted.get("cycles_since_last_candidate", 0),
                persisted.get("cycles_since_last_trade", 0),
                len(persisted.get("tunings") or []),
            )
        return autotune

    def _build_news_pipeline(self) -> tuple[CandidateStore | None, NewsScheduler | None]:
        """Construct the news aggregator + scheduler (or short-circuit).

        The pipeline is *always* constructed in Phase 2 because the
        universe builder consumes the candidate store. Disabling news
        is done by clearing ``enabled_sources``, not by skipping the
        pipeline entirely — that way the universe builder still has a
        valid (empty) store to read.
        """
        store, _aggregator, scheduler = build_news_pipeline(
            self.cfg.news,
            project_root=self._project_root,
            instrument_cache=self.cache,
            logger=get_logger("news", tag="news"),
        )
        if not self.cfg.news.enabled:
            self.log.info(
                "[news] disabled in config — universe will only build from "
                "previously persisted candidates and LLM rotation",
            )
        return store, scheduler

    def _build_fundamentals_cache(self) -> FundamentalsCache | None:
        """Wire the yfinance-backed fundamentals cache (Phase 3).

        Returns ``None`` when ``[fundamentals] enabled = false`` so the
        cycle and Telegram surfaces fall back to the Phase 2 behaviour
        (universe + news only, no per-symbol fundamentals).
        """
        cache = build_fundamentals_cache(
            self.cfg.fundamentals,
            project_root=self._project_root,
            logger=get_logger("fundamentals", tag="fund"),
        )
        if cache is None:
            self.log.info("[fundamentals] disabled in config")
            return None
        self.log.info(
            "[fundamentals] cache ready (%d entries at %s)",
            len(cache), cache.path,
        )
        return cache

    def _build_earnings_calendar(self) -> "EarningsCalendarCache | None":
        """Wire the yfinance-backed earnings-date cache.

        Returns ``None`` when ``[earnings_calendar] enabled = false``
        so the proximity tool and pre-earnings rules become no-ops.
        """
        cfg = self.cfg.earnings_calendar
        if not cfg.enabled:
            self.log.info("[earnings] calendar disabled in config")
            return None
        from .strategy.earnings_calendar import EarningsCalendarCache

        cache = EarningsCalendarCache(
            self._project_root / cfg.cache_path,
            ttl_seconds=int(cfg.ttl_hours * 3600),
            logger=get_logger("earnings", tag="earn"),
        )
        self.log.info(
            "[earnings] calendar ready (%d cached entries at %s)",
            len(cache.snapshot()), self._project_root / cfg.cache_path,
        )
        return cache

    def _build_alert_hub(self) -> AlertHub | None:
        """Build the alert fan-out hub if any chat IDs are allow-listed."""
        if not self.cfg.alerting.is_enabled:
            self.log.info(
                "[alerts] no allowed chats configured (TELEGRAM_ALLOWED_CHAT_IDS empty) — "
                "alerts disabled. Trading bot continues normally."
            )
            return None
        subs_path = self._project_root / self.cfg.alerting.subscriptions_file
        subs = AlertSubscriptions(
            subs_path,
            default_set=safety_only_default(),
            logger=get_logger("alerts.subs", tag="alerts"),
        )
        hub = AlertHub(
            allowed_chat_ids=self.cfg.alerting.allowed_chat_ids,
            subscriptions=subs,
            max_per_chat=self.cfg.alerting.max_queue_per_chat,
            logger=get_logger("alerts.hub", tag="alerts"),
        )
        self.log.info(
            "[alerts] hub ready (chats=%s, subs=%s)",
            ", ".join(str(c) for c in self.cfg.alerting.allowed_chat_ids),
            subs_path,
        )
        return hub

    def _build_tool_orchestrator(self) -> ToolOrchestrator | None:
        if not self.cfg.tools.enabled:
            self.log.info("[tools] extended catalog disabled via config")
            return None
        return ToolOrchestrator(
            cfg=self.cfg,
            etoro=self._etoro,
            logger=get_logger("strategy.tools", tag="tools"),
        )

    def _build_usage_tracker(self) -> LLMUsageTracker | None:
        """Construct the LLM usage tracker if AI is available.

        Falls back gracefully when AI isn't configured — `/tokens`
        will then report ``enabled=false`` and the Telegram surface
        explains why.
        """
        if not self.cfg.azure.is_configured:
            return None
        return LLMUsageTracker(
            self._project_root / "data" / "llm_usage.jsonl",
            deployment=self.cfg.azure.deployment or "",
            logger=get_logger("ai.usage_tracker", tag="ai"),
        )

    def _build_ai_client(self) -> AzureFoundryClient | None:
        if not self.cfg.ai.enabled:
            self.log.warning("AI disabled in config — running deterministic-only")
            return None
        if not self.cfg.azure.is_configured:
            self.log.warning("Azure credentials missing — running deterministic-only")
            return None
        try:
            client = AzureFoundryClient(
                self.cfg.azure,
                max_completion_tokens=self.cfg.ai.max_completion_tokens,
                logger=get_logger("ai.azure", tag="ai"),
            )
            self.log.info(
                "Azure Foundry client ready (deployment=%s, reasoning=%s)",
                self.cfg.azure.deployment, self.cfg.azure.is_reasoning_model,
            )
            return client
        except AzureUnavailable as exc:
            self.log.warning("Azure unavailable: %s — running deterministic-only", exc)
            return None

    def _maybe_start_control_server(self) -> ControlHTTPServer | None:
        ctrl_cfg = self.cfg.control
        if not ctrl_cfg.enabled:
            self.log.info("[control] HTTP server disabled by config")
            return None
        if not ctrl_cfg.is_secured:
            self.log.warning(
                "[control] INTERNAL_API_TOKEN not set — refusing to expose control "
                "server. Set one in .env to enable Telegram /pause /resume /panic.")
            return None
        try:
            server = ControlHTTPServer(
                host=ctrl_cfg.host,
                port=ctrl_cfg.port,
                bearer_token=ctrl_cfg.internal_api_token or "",
                controller=self.controller,
                logger=get_logger("control.server", tag="control"),
            )
            server.start()
            return server
        except OSError as exc:
            self.log.error("[control] failed to bind %s:%d — %s", ctrl_cfg.host, ctrl_cfg.port, exc)
            return None

    def _restore_persisted_state(self) -> None:
        loaded, meta = self._persistence.load()
        self._initial_paused = False
        if loaded is not None:
            self.state = loaded
            if meta is not None and meta.paused:
                self._initial_paused = True
            # Don't carry the previous process' cycle counter into logs.
            self.log.info(
                "[state] restored: bot_owned=%d, cycle_count=%d, paused=%s",
                len(self.state.bot_owned_positions),
                self.state.cycle_count,
                bool(meta and meta.paused),
            )

    def _announce_identity(self) -> None:
        try:
            identity = fetch_identity(self._etoro)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("identity fetch failed: %s", exc)
            return
        self.log.info(
            "identity: gcid=%s realCid=%s demoCid=%s",
            identity.gcid, identity.real_cid, identity.demo_cid,
        )

    def _shutdown(self) -> None:
        try:
            self.controller.persist_state()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.cache.save()
        except OSError:
            pass
        if self._control_server is not None:
            self._control_server.stop()
        self._etoro.close()
        if self._config_store is not None:
            self._config_store.close()

    def _interruptible_sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop and not self.controller.stop_requested and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001 - kept for symmetry
    config_store = open_store(
        PROJECT_ROOT / DEFAULT_CONFIG_DB_PATH,
        logger=get_logger("config_store", tag="config"),
    )
    try:
        cfg = load_config(config_store=config_store)
    except Exception as exc:  # noqa: BLE001
        config_store.close()
        print(f"[fatal] config error: {exc}", file=sys.stderr)
        return 2
    bot = TradingBot(cfg, config_store=config_store)
    return bot.run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
