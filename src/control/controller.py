"""Thread-safe controller mediating Telegram commands and the trading loop.

The trading loop runs in the main thread; the HTTP server (which
serves Telegram) runs in a daemon thread. Both touch the same shared
state (``BotState``), the same eToro client (sync, thread-unsafe), and
the same in-memory guardrails. The controller serializes them with a
single re-entrant lock that the loop holds during a cycle and the
handlers hold while running an action.

Responsibilities:

- ``pause()`` / ``resume()``: flip a flag the loop checks each tick.
- ``panic_close_all()``: close every open position right now (regardless
  of who opened it) using the same TradeExecutor the cycle uses.
- ``apply_guardrails_change()``: edit a [guardrails] field at runtime.
- ``ask()``: forward a free-text question to the LLM with a snapshot
  of bot state included as context.
- ``snapshot_*()``: read views (status, portfolio, history, universe).
- ``persist_state()``: save BotState + ``paused`` to disk.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

from ..ai.azure_client import AzureFoundryClient, AzureUnavailable
from ..ai.prompts import build_qa_prompt
from ..alerts import AlertHub, AlertType
from ..config import AppConfig, GuardrailsConfig
from ..config_store import ConfigStore
from ..etoro.client import EtoroClient
from ..etoro.errors import EtoroApiError
from ..etoro.trading import close_position_by_market, fetch_portfolio
from ..news.channel_probe import probe_many
from ..persistence import StatePersistence
from ..state import BotState
from ..strategy.directives import (
    DirectiveError,
    Directives,
    DirectivesStore,
    NOTES_MAX_CHARS,
    STRUCTURED_KEYS,
)
from ..strategy.rules_summary import ToolDescription, build_rules_payload
from ..telemetry import TelemetryStore
from ..trade_history import TradeHistoryEntry, TradeHistoryLog, utc_now_iso


class ControllerError(Exception):
    """Raised by controller actions for caller-visible failures."""


@dataclass(frozen=True)
class ControllerStatus:
    paused: bool
    cycle_count: int
    halted_today: bool
    halted_day: str | None
    started_at_unix: float
    last_cycle_started_unix: float | None
    last_cycle_finished_unix: float | None
    last_error: str | None
    bot_owned_position_count: int
    tracked_count: int
    base_count: int
    llm_count: int
    trading_mode: str
    env_segment: str
    ai_enabled: bool


class BotController:
    """Single source of truth for cross-thread bot operations."""

    def __init__(
        self,
        *,
        cfg: AppConfig,
        state: BotState,
        etoro: EtoroClient,
        ai_client: AzureFoundryClient | None,
        telemetry: TelemetryStore,
        history: TradeHistoryLog,
        persistence: StatePersistence,
        config_store: ConfigStore | None = None,
        alerts: AlertHub | None = None,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._etoro = etoro
        self._ai = ai_client
        self._telemetry = telemetry
        self._history = history
        self._persistence = persistence
        self._config_store = config_store
        self._alerts = alerts
        self._log = logger or logging.getLogger("etrader.control.controller")
        self._lock = threading.RLock()
        self._paused = False
        self._stop_requested = False
        self._tool_orchestrator: Any | None = None
        self._news_store: Any | None = None
        self._news_scheduler: Any | None = None
        self._fundamentals: Any | None = None
        self._autotune_state: Any | None = None
        self._performance: Any | None = None
        self._dynamic_stops: Any | None = None
        self._directives: DirectivesStore | None = None
        self._token_usage: Any | None = None

    # ------------------------------------------------------------------
    # Lock + pause primitives the cycle loop uses
    # ------------------------------------------------------------------

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def request_stop(self) -> None:
        """Ask the trading loop to exit (full process shutdown)."""
        self._stop_requested = True

    def set_tool_orchestrator(self, orchestrator: Any) -> None:
        """Wire in the live ToolOrchestrator so /signals can introspect it."""
        with self._lock:
            self._tool_orchestrator = orchestrator

    def set_news_store(self, store: Any, scheduler: Any) -> None:
        """Wire in the live news pipeline so /news can introspect it."""
        with self._lock:
            self._news_store = store
            self._news_scheduler = scheduler

    def set_fundamentals(self, cache: Any) -> None:
        """Wire in the live fundamentals cache so /fundamentals can read it."""
        with self._lock:
            self._fundamentals = cache

    def set_autotune_state(self, autotune_state: Any) -> None:
        """Wire the autonomous-tuner state so persist_state can checkpoint it."""
        with self._lock:
            self._autotune_state = autotune_state

    def set_performance_tracker(self, tracker: Any) -> None:
        """Wire the performance tracker so /stats and /ask can read it."""
        with self._lock:
            self._performance = tracker

    def set_dynamic_stops(self, store: Any) -> None:
        """Wire the per-position SL/TP store so persist_state can checkpoint it."""
        with self._lock:
            self._dynamic_stops = store

    def set_directives_store(self, store: DirectivesStore) -> None:
        """Wire the operator-directives store so Telegram can edit it."""
        with self._lock:
            self._directives = store

    def set_token_usage_tracker(self, tracker: Any) -> None:
        """Wire the LLM-usage tracker so /tokens can read its rollups."""
        with self._lock:
            self._token_usage = tracker

    def snapshot_performance(self) -> dict[str, Any]:
        """Return the structured performance payload for /stats and /ask.

        Returns ``{"enabled": False}`` when the tracker isn't wired
        (e.g. unit tests). All concrete fields live inside
        :meth:`PerformanceTracker.summary` — see that for the schema.
        """
        with self._lock:
            tracker = getattr(self, "_performance", None)
        if tracker is None:
            return {"enabled": False}
        payload = tracker.summary()
        payload["enabled"] = True
        return payload

    def snapshot_performance_symbols(self) -> dict[str, Any]:
        """Per-symbol breakdown for ``/stats by-symbol``."""
        with self._lock:
            tracker = getattr(self, "_performance", None)
        if tracker is None:
            return {"enabled": False, "rows": []}
        return {"enabled": True, "rows": tracker.by_symbol()}

    def snapshot_performance_dailies(self, *, limit: int = 30) -> dict[str, Any]:
        """Recent daily snapshots — newest last."""
        with self._lock:
            tracker = getattr(self, "_performance", None)
        if tracker is None:
            return {"enabled": False, "rows": []}
        return {
            "enabled": True,
            "rows": [d.to_dict() for d in tracker.daily_history(limit=limit)],
        }

    def snapshot_performance_closed_trades(
        self, *, limit: int = 50,
    ) -> dict[str, Any]:
        with self._lock:
            tracker = getattr(self, "_performance", None)
        if tracker is None:
            return {"enabled": False, "rows": []}
        return {
            "enabled": True,
            "rows": [t.to_dict() for t in tracker.closed_trades(limit=limit)],
        }

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def pause(self, *, reason: str | None = None) -> dict[str, Any]:
        with self._lock:
            already = self._paused
            self._paused = True
            self.persist_state()
        self._log.info("[control] pause requested%s", f" — {reason}" if reason else "")
        if not already:
            self._emit_alert(
                AlertType.BOT_PAUSED_RESUMED,
                title="BOT PAUSED",
                body=f"Reason: {reason}" if reason else "",
            )
        return {"paused": True, "was_already_paused": already, "reason": reason}

    def resume(self) -> dict[str, Any]:
        with self._lock:
            already = not self._paused
            self._paused = False
            self.persist_state()
        self._log.info("[control] resume requested")
        if not already:
            self._emit_alert(
                AlertType.BOT_PAUSED_RESUMED,
                title="BOT RESUMED",
                body="Trading loop is running again.",
            )
        return {"paused": False, "was_already_running": already}

    def unhalt(self, *, reason: str | None = None) -> dict[str, Any]:
        """Force-clear the daily-loss kill switch immediately.

        Use case: the operator changed ``daily_loss_stop_usd`` or
        explicitly wants the bot to resume mid-day without waiting for
        the UTC rollover. We clear ``halted_today`` and rebase the
        equity baseline against the current portfolio so any
        subsequent drawdown check starts from the resume point — NOT
        from the morning baseline that already counted today's losses.

        Returns the previous halt state so callers can format an
        appropriate "was already running" / "halt cleared" message.
        """
        with self._lock:
            was_halted = bool(self._state.halted_today)
            self._state.halted_today = False
            # Force the next kill-switch tick to rebase against live
            # equity — wiping ``session_baseline_equity`` triggers the
            # ``current_equity is not None`` branch in the kill switch.
            self._state.session_baseline_equity = None
            self.persist_state()
        self._log.info(
            "[control] unhalt requested%s", f" — {reason}" if reason else "",
        )
        if was_halted:
            self._emit_alert(
                AlertType.BOT_PAUSED_RESUMED,
                title="BOT UNHALTED",
                body=(
                    "Daily-loss kill switch cleared manually. "
                    "Equity baseline will rebase on the next cycle."
                ),
            )
        return {"was_halted": was_halted, "reason": reason}

    def set_paused_initial(self, paused: bool) -> None:
        """Restore paused flag at startup from persisted state."""
        with self._lock:
            self._paused = bool(paused)

    # ------------------------------------------------------------------
    # Panic close
    # ------------------------------------------------------------------

    def panic_close_all(
        self,
        *,
        scope: str = "all",
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Close every open position. Pauses the loop first.

        ``scope``:
        - ``"all"`` (default): close every open position on the account
          including ones the user opened manually outside the bot.
        - ``"bot_owned"``: close only positions in
          ``state.bot_owned_positions``.

        Calls eToro under :attr:`lock`; safe to invoke from the HTTP
        thread while a cycle is sleeping or running (it'll wait).
        """
        scope_norm = (scope or "all").lower().strip()
        if scope_norm not in {"all", "bot_owned"}:
            raise ControllerError(f"invalid scope {scope!r} (use 'all' or 'bot_owned')")

        with self._lock:
            self._paused = True
            try:
                snapshot = fetch_portfolio(self._etoro, self._cfg.env_segment)
            except EtoroApiError as exc:
                raise ControllerError(f"portfolio fetch failed: {exc}") from exc

            target_positions = []
            for p in snapshot.positions:
                if p.mirror_id:
                    # Mirror positions are tied to a copy-trade and can't
                    # be closed individually via this endpoint.
                    continue
                if scope_norm == "bot_owned" and p.position_id not in self._state.bot_owned_positions:
                    continue
                target_positions.append(p)

            results: list[dict[str, Any]] = []
            for p in target_positions:
                try:
                    close_position_by_market(
                        self._etoro,
                        env=self._cfg.env_segment,
                        position_id=p.position_id,
                        instrument_id=p.instrument_id,
                    )
                    status = "ok"
                    detail = "closed"
                except EtoroApiError as exc:
                    status = "failed"
                    detail = str(exc)
                self._state.remove_owned(p.position_id)
                self._state.mark_action(p.instrument_id)
                entry = TradeHistoryEntry(
                    timestamp=utc_now_iso(),
                    action="CLOSE",
                    status="panic_close" if status == "ok" else "failed",
                    symbol=str(p.instrument_id),
                    instrument_id=p.instrument_id,
                    amount_usd=p.amount,
                    order_id=None,
                    position_id=p.position_id,
                    detail=f"panic({scope_norm})" + ("" if status == "ok" else f": {detail}"),
                )
                self._history.append(entry)
                results.append({
                    "position_id": p.position_id,
                    "instrument_id": p.instrument_id,
                    "status": status,
                    "detail": detail,
                })
                # eToro caps trade execution at 20 req/min (~3s spacing).
                time.sleep(self._cfg.operations.trade_spacing_seconds)

            self.persist_state()

        closed_ok = sum(1 for r in results if r["status"] == "ok")
        self._log.warning(
            "[control] PANIC scope=%s closed=%d reason=%s",
            scope_norm, closed_ok, reason,
        )
        self._emit_alert(
            AlertType.PANIC_CLOSE,
            title=f"PANIC CLOSE ({scope_norm})",
            body=(
                f"Closed {closed_ok}/{len(results)} positions. "
                f"Reason: {reason}" if reason else
                f"Closed {closed_ok}/{len(results)} positions."
            ),
        )
        return {
            "scope": scope_norm,
            "closed_attempted": len(results),
            "closed_ok": closed_ok,
            "results": results,
            "now_paused": True,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Live config updates (guardrails only, by design)
    # ------------------------------------------------------------------

    GUARDRAIL_FIELDS = (
        "max_per_trade_usd",
        "max_parallel_trades",
        "daily_loss_stop_usd",
        "per_instrument_cooldown_min",
        "default_stop_loss_pct",
        "default_take_profit_pct",
        "max_leverage",
        "max_bot_invested_usd",
        "min_amend_remainder_usd",
        "pre_earnings_buy_blackout_hours",
    )

    @staticmethod
    def _coerce_guardrail(field_name: str, raw: Any) -> Any:
        spec = {f.name: f.type for f in fields(GuardrailsConfig)}
        if field_name not in spec:
            raise ControllerError(f"unknown guardrails field: {field_name}")
        try:
            if spec[field_name] in (int, "int"):
                return int(float(raw))
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"value for {field_name} must be numeric: {raw!r}") from exc

    def get_guardrails(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._cfg.guardrails)

    def apply_guardrails_change(self, key: str, value: Any) -> dict[str, Any]:
        if key not in self.GUARDRAIL_FIELDS:
            raise ControllerError(
                f"unknown guardrails key: {key!r}. Allowed: {', '.join(self.GUARDRAIL_FIELDS)}"
            )
        coerced = self._coerce_guardrail(key, value)
        with self._lock:
            previous = getattr(self._cfg.guardrails, key)
            setattr(self._cfg.guardrails, key, coerced)
            self._persist_config_field("guardrails", key, coerced)
        self._log.info("[control] guardrails.%s: %r -> %r", key, previous, coerced)
        return {"key": key, "previous": previous, "current": coerced}

    def _persist_config_field(self, section: str, key: str, value: Any) -> None:
        """Write a single config override through to the SQLite store.

        Swallows store errors so a transient DB failure can't kill the
        Telegram edit — the in-memory dataclass is the source of truth
        for the running process and will be re-synced on the next save.
        """
        if self._config_store is None:
            return
        try:
            self._config_store.set_field(section, key, value)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("[control] config persist failed (%s.%s): %s", section, key, exc)

    # ------------------------------------------------------------------
    # Operator directives (persistent rules surfaced to the LLM)
    # ------------------------------------------------------------------

    def snapshot_directives(self) -> dict[str, Any]:
        """Return the live directives snapshot for /directives + LLM context."""
        with self._lock:
            store = self._directives
        if store is None:
            return {
                "enabled": False,
                "structured_keys": list(STRUCTURED_KEYS),
                "notes_max_chars": NOTES_MAX_CHARS,
                "values": Directives().to_dict(),
            }
        return {
            "enabled": True,
            "structured_keys": list(STRUCTURED_KEYS),
            "notes_max_chars": NOTES_MAX_CHARS,
            "values": store.to_dict(),
        }

    def set_directive(self, key: str, value: Any) -> dict[str, Any]:
        store = self._require_directives_store()
        try:
            previous, current = store.set_field(key, value)
        except DirectiveError as exc:
            raise ControllerError(str(exc)) from exc
        with self._lock:
            self.persist_state()
        return {"key": key, "previous": previous, "current": current}

    def clear_directive(self, key: str) -> dict[str, Any]:
        store = self._require_directives_store()
        try:
            previous, current = store.clear_field(key)
        except DirectiveError as exc:
            raise ControllerError(str(exc)) from exc
        with self._lock:
            self.persist_state()
        return {"key": key, "previous": previous, "current": current}

    def set_directive_note(self, text: str) -> dict[str, Any]:
        store = self._require_directives_store()
        previous, current = store.set_notes(text or "")
        with self._lock:
            self.persist_state()
        return {"previous": previous, "current": current}

    def clear_directive_note(self) -> dict[str, Any]:
        store = self._require_directives_store()
        previous = store.clear_notes()
        with self._lock:
            self.persist_state()
        return {"previous": previous, "current": ""}

    def _require_directives_store(self) -> DirectivesStore:
        with self._lock:
            store = self._directives
        if store is None:
            raise ControllerError("directives store not wired (server starting up)")
        return store

    # ------------------------------------------------------------------
    # LLM token usage
    # ------------------------------------------------------------------

    def snapshot_token_usage(self) -> dict[str, Any]:
        """Return token / cost rollups for /tokens.

        Falls back to ``{"enabled": False}`` when the usage tracker
        isn't wired (e.g. AI disabled). The concrete shape lives in
        :meth:`src.ai.usage_tracker.LLMUsageTracker.snapshot`.
        """
        with self._lock:
            tracker = self._token_usage
        if tracker is None:
            return {"enabled": False}
        payload = tracker.snapshot()
        payload["enabled"] = True
        return payload

    # ------------------------------------------------------------------
    # AI Q&A
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ControllerError("question must not be empty")
        if self._ai is None:
            raise ControllerError(
                "AI is not configured (Azure OpenAI not set up). "
                "I can still answer status/portfolio/history via dedicated commands."
            )

        snapshot = self.snapshot_status_dict()
        snapshot["guardrails"] = self.get_guardrails()
        snapshot["recent_history"] = [
            e.to_dict() for e in self._history.tail(limit=20)
        ]
        # Merge in the telemetry payload (positions, equity, etc.) so
        # the prompt's partitioning of bot_owned vs manual positions
        # has the source arrays it needs.
        snapshot.update(self._telemetry.snapshot())
        rules_payload = self.snapshot_strategy_rules()
        # Authoritative bot-attributable P/L for the LLM — without
        # this, the LLM hand-rolls numbers from trade_history.jsonl
        # and hallucinates (see git history for the "$450 loss"
        # incident). With it, every P/L number is pre-computed.
        performance_payload: dict[str, Any] | None = None
        if self._performance is not None:
            try:
                performance_payload = self._performance.summary()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("performance.summary() failed: %s", exc)

        directives_payload = self.snapshot_directives()
        system, user = build_qa_prompt(
            question=question,
            bot_snapshot=snapshot,
            strategy_rules=rules_payload,
            performance=performance_payload,
            directives=directives_payload,
        )
        try:
            result = self._ai.chat_json(
                system=system, user=user, require_json=False, call_type="qa",
            )
        except AzureUnavailable as exc:
            raise ControllerError(f"LLM call failed: {exc}") from exc
        return {
            "question": question,
            "answer": result.text or "(no answer)",
            "latency_ms": result.latency_ms,
        }

    # ------------------------------------------------------------------
    # Strategy rules introspection
    # ------------------------------------------------------------------

    def snapshot_strategy_rules(self) -> dict[str, Any]:
        """Build a structured snapshot of the live rule set + tool catalog."""
        with self._lock:
            tools_desc: list[ToolDescription] = []
            tool_perf: list[dict[str, Any]] = []
            if self._tool_orchestrator is not None:
                for d in self._tool_orchestrator.registry.descriptions():
                    tools_desc.append(ToolDescription(
                        name=str(d["name"]),
                        family=str(d["family"]),
                        purpose=str(d["purpose"]),
                        role=str(d["role"]),
                        asset_classes=tuple(d["asset_classes"]),
                    ))
                tool_perf = [s.to_dict() for s in self._tool_orchestrator.performance.all_stats()]
            payload = build_rules_payload(
                strategy=self._cfg.strategy,
                guardrails=self._cfg.guardrails,
                tools=tools_desc,
            )
            payload["tool_performance"] = tool_perf
            return payload

    # ------------------------------------------------------------------
    # Read views
    # ------------------------------------------------------------------

    def snapshot_status(self) -> ControllerStatus:
        d = self._telemetry.snapshot()
        return ControllerStatus(
            paused=self._paused,
            cycle_count=int(d.get("cycle_count") or 0),
            halted_today=bool(self._state.halted_today),
            halted_day=self._state.halted_day,
            started_at_unix=float(self._state.started_at),
            last_cycle_started_unix=d.get("last_cycle_started_unix"),
            last_cycle_finished_unix=d.get("last_cycle_finished_unix"),
            last_error=d.get("last_error"),
            bot_owned_position_count=len(self._state.bot_owned_positions),
            tracked_count=len(d.get("tracked_instrument_ids") or []),
            base_count=int(d.get("base_count") or 0),
            llm_count=int(d.get("llm_count") or 0),
            trading_mode=self._cfg.trading_mode,
            env_segment=self._cfg.env_segment,
            ai_enabled=bool(self._cfg.ai.enabled and self._cfg.azure.is_configured),
        )

    def snapshot_status_dict(self) -> dict[str, Any]:
        return asdict(self.snapshot_status())

    def snapshot_portfolio(self) -> dict[str, Any]:
        d = self._telemetry.snapshot()
        return {
            "summary": dict(d.get("portfolio_summary") or {}),
            "positions": list(d.get("portfolio_positions") or []),
            "bot_owned_position_ids": list(d.get("bot_owned_position_ids") or []),
            "paused": self._paused,
        }

    def snapshot_universe(self) -> dict[str, Any]:
        d = self._telemetry.snapshot()
        return {
            "instrument_ids": list(d.get("tracked_instrument_ids") or []),
            "symbols": list(d.get("tracked_symbols") or []),
            "base_count": int(d.get("base_count") or 0),
            "llm_count": int(d.get("llm_count") or 0),
            "reasons": dict(d.get("universe_reasons") or {}),
            "source_counts": dict(d.get("universe_source_counts") or {}),
            "rejected": dict(d.get("universe_rejected") or {}),
        }

    def snapshot_fundamentals(self, *, symbol: str | None = None) -> dict[str, Any]:
        """Return the cached fundamentals payload.

        When ``symbol`` is supplied, returns ``{"symbol": SYM,
        "snapshot": {...}}`` (or ``{"symbol": SYM, "snapshot": None}``
        if we have nothing cached). When ``symbol`` is None, returns a
        summary view ``{"count": N, "items": [{"symbol", "name",
        "sector", "fetched_at_unix"}, ...]}`` so /fundamentals (with no
        argument) can show what's currently cached.

        Safe to call when the cache hasn't been wired: returns
        ``{"enabled": false, ...}``.
        """
        if self._fundamentals is None:
            return {"enabled": False, "symbol": symbol, "snapshot": None, "items": []}
        if symbol:
            sym = symbol.strip().upper()
            snap = self._fundamentals.get(sym)
            return {
                "enabled": True,
                "symbol": sym,
                "snapshot": snap.to_dict() if snap is not None else None,
            }
        items: list[dict[str, Any]] = []
        for s in self._fundamentals.all().values():
            items.append({
                "symbol": s.symbol,
                "name": s.name,
                "sector": s.sector,
                "industry": s.industry,
                "quote_type": s.quote_type,
                "fetched_at_unix": s.fetched_at_unix,
                "next_earnings_unix": s.next_earnings_unix,
            })
        items.sort(key=lambda i: i["symbol"])
        return {"enabled": True, "count": len(items), "items": items}

    def snapshot_news(self, *, limit: int = 25) -> dict[str, Any]:
        """Return the top-N candidates from the news store plus the last scan stats.

        Safe to call when the news pipeline hasn't been wired (returns
        an empty payload). ``limit`` is clamped to [1, 200].
        """
        limit = max(1, min(int(limit or 25), 200))
        if self._news_store is None:
            return {"candidates": [], "last_scan": None, "next_scan_in_seconds": None}
        candidates = self._news_store.top(limit)
        last_scan = self._last_scan_dict()
        next_scan = None
        if self._news_scheduler is not None:
            next_scan = self._news_scheduler.seconds_until_next_run()
        return {
            "candidates": [
                {
                    "symbol": c.symbol,
                    "score": round(c.score, 4),
                    "sources": list(c.sources),
                    "headlines": list(c.headlines),
                    "first_seen_unix": c.first_seen_unix,
                    "last_seen_unix": c.last_seen_unix,
                    "reason": c.reason,
                }
                for c in candidates
            ],
            "last_scan": last_scan,
            "next_scan_in_seconds": next_scan,
        }

    def snapshot_news_channels(self) -> dict[str, Any]:
        """Overview of every configured news source / channel.

        Each entry combines:

        * ``enabled``       — whether the source name appears in
          ``[news] enabled_sources`` (config intent);
        * ``wired``         — whether the aggregator actually constructed
          a plug-in for that name (config can list unknown sources);
        * ``disabled_reason`` — self-reported disabled reason (e.g. SEC
          EDGAR with no ``SEC_USER_AGENT``);
        * ``weight``        — effective per-source weight multiplier
          used by the scoring layer;
        * ``last_scan``     — items_kept / error from the most recent
          aggregator run for this source.

        Plus a top-level ``last_scan`` block (same shape as ``/news``)
        and the seconds until the next scheduled scan.
        """
        with self._lock:
            scheduler = self._news_scheduler
            aggregator = scheduler.aggregator if scheduler is not None else None
            news_cfg = self._cfg.news

        configured = [str(s).strip().lower() for s in news_cfg.enabled_sources if s]
        wired_map: dict[str, Any] = {}
        if aggregator is not None:
            for src in aggregator.sources:
                name = str(getattr(src, "name", src.__class__.__name__)).lower()
                wired_map[name] = src

        last_scan = self._last_scan_dict()
        per_source_counts = (last_scan or {}).get("per_source_counts") or {}
        per_source_errors = (last_scan or {}).get("per_source_errors") or {}

        names: list[str] = []
        for n in configured:
            if n not in names:
                names.append(n)
        for n in wired_map.keys():
            if n not in names:
                names.append(n)

        channels: list[dict[str, Any]] = []
        for name in names:
            src = wired_map.get(name)
            disabled_reason = (
                str(getattr(src, "_disabled_reason", "") or "") if src is not None else ""
            ) or None
            weight = aggregator.source_weight(name) if aggregator is not None else None
            channels.append({
                "name": name,
                "enabled": name in configured,
                "wired": src is not None,
                "class": src.__class__.__name__ if src is not None else None,
                "disabled_reason": disabled_reason,
                "weight": weight,
                "last_items_kept": int(per_source_counts.get(name, 0) or 0),
                "last_error": per_source_errors.get(name) or None,
            })

        next_scan = None
        if scheduler is not None:
            next_scan = scheduler.seconds_until_next_run()

        return {
            "pipeline_enabled": bool(news_cfg.enabled),
            "scan_interval_minutes": int(news_cfg.scan_interval_minutes),
            "ttl_hours": int(news_cfg.ttl_hours),
            "half_life_hours": float(news_cfg.half_life_hours),
            "channels": channels,
            "last_scan": last_scan,
            "next_scan_in_seconds": next_scan,
        }

    def test_news_channels(
        self,
        *,
        only: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Run a side-effect-free dry-run against each configured source.

        Results are not folded into the candidate store — this is a
        health probe, not an ingest path. ``only`` filters by source
        name (case-insensitive); unknown names are dropped silently so
        the caller can pass a user-supplied list verbatim.

        Returns a payload with one :class:`ChannelProbeResult`-shaped
        dict per probed source plus a tiny summary the renderer can
        show first.
        """
        with self._lock:
            scheduler = self._news_scheduler
            aggregator = scheduler.aggregator if scheduler is not None else None

        if aggregator is None:
            return {
                "available": False,
                "results": [],
                "summary": {"probed": 0, "ok": 0, "failed": 0},
            }

        known: list[str] = []
        if self._news_store is not None:
            try:
                known = [c.symbol for c in self._news_store.top(50)]
            except Exception:  # noqa: BLE001 — never let cache shape kill the probe
                known = []

        results = probe_many(aggregator.sources, only=only, known_symbols=known)
        ok_count = sum(1 for r in results if r.ok)
        return {
            "available": True,
            "summary": {
                "probed": len(results),
                "ok": ok_count,
                "failed": len(results) - ok_count,
            },
            "results": [r.to_dict() for r in results],
        }

    def _last_scan_dict(self) -> dict[str, Any] | None:
        """Shared helper: render the aggregator's most-recent run stats."""
        if self._news_scheduler is None:
            return None
        stats = self._news_scheduler.last_stats
        if stats is None:
            return None
        return {
            "started_at_unix": stats.started_at_unix,
            "finished_at_unix": stats.finished_at_unix,
            "items_fetched": stats.items_fetched,
            "items_kept": stats.items_kept,
            "observations_recorded": stats.observations_recorded,
            "per_source_counts": dict(stats.per_source_counts),
            "per_source_errors": dict(stats.per_source_errors),
        }

    def recent_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 20), 200))
        return [e.to_dict() for e in self._history.tail(limit=limit)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_state(self) -> None:
        autotune_payload: dict[str, Any] | None = None
        if self._autotune_state is not None:
            try:
                autotune_payload = self._autotune_state.to_dict()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("[control] autotune snapshot failed: %s", exc)
        dynamic_stops_payload: dict[str, Any] | None = None
        if self._dynamic_stops is not None:
            try:
                dynamic_stops_payload = self._dynamic_stops.to_persistable()
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "[control] dynamic-stops snapshot failed: %s", exc,
                )
        directives_payload: dict[str, Any] | None = None
        if self._directives is not None:
            try:
                directives_payload = self._directives.to_persistable()
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "[control] directives snapshot failed: %s", exc,
                )
        try:
            self._persistence.save(
                self._state,
                paused=self._paused,
                autotune_payload=autotune_payload,
                dynamic_stops_payload=dynamic_stops_payload,
                directives_payload=directives_payload,
            )
        except Exception as exc:  # noqa: BLE001 - never let save kill us
            self._log.warning("[control] persist failed: %s", exc)

    # ------------------------------------------------------------------
    # Alerts (Telegram /alerts surface)
    # ------------------------------------------------------------------

    @property
    def alerts(self) -> AlertHub | None:
        return self._alerts

    def _emit_alert(
        self,
        alert_type: AlertType,
        *,
        title: str,
        body: str = "",
    ) -> None:
        """Best-effort alert emit. Never raises; always swallows."""
        if self._alerts is None:
            return
        try:
            self._alerts.emit(alert_type, title=title, body=body)
        except Exception as exc:  # noqa: BLE001 - alerts never block trading
            self._log.warning("[alerts] emit failed for %s: %s", alert_type.value, exc)

    def emit_alert(self, type_str: str, *, title: str, body: str = "") -> None:
        """External-friendly emitter (string type → AlertType)."""
        resolved = AlertType.from_value(type_str)
        if resolved is None:
            return
        self._emit_alert(resolved, title=title, body=body)

    def list_alert_types(self) -> list[dict[str, Any]]:
        """Return every available alert type with a human label."""
        return [
            {"type": t.value, "label": _ALERT_LABELS.get(t, t.value)}
            for t in AlertType.all_types()
        ]

    def _require_alerts(self) -> AlertHub:
        if self._alerts is None:
            raise ControllerError("alerts are disabled (no allowed chats configured)")
        return self._alerts

    def get_alert_subscriptions(self, chat_id: int) -> dict[str, Any]:
        hub = self._require_alerts()
        enabled = hub.subscriptions.enabled_for(chat_id)
        return {
            "chat_id": int(chat_id),
            "enabled": sorted(t.value for t in enabled),
            "available": [
                {
                    "type": t.value,
                    "label": _ALERT_LABELS.get(t, t.value),
                    "enabled": t in enabled,
                }
                for t in AlertType.all_types()
            ],
        }

    def set_alert_subscription(
        self, chat_id: int, type_str: str, enabled: bool,
    ) -> dict[str, Any]:
        hub = self._require_alerts()
        resolved = AlertType.from_value(type_str)
        if resolved is None:
            raise ControllerError(
                f"unknown alert type {type_str!r}. "
                f"Allowed: {', '.join(t.value for t in AlertType.all_types())}"
            )
        new_set = hub.subscriptions.set_enabled(chat_id, resolved, bool(enabled))
        return {
            "chat_id": int(chat_id),
            "type": resolved.value,
            "enabled": bool(enabled),
            "all_enabled": sorted(t.value for t in new_set),
        }

    def toggle_alert_subscription(
        self, chat_id: int, type_str: str,
    ) -> dict[str, Any]:
        hub = self._require_alerts()
        resolved = AlertType.from_value(type_str)
        if resolved is None:
            raise ControllerError(f"unknown alert type {type_str!r}")
        new_state, full_set = hub.subscriptions.toggle(chat_id, resolved)
        return {
            "chat_id": int(chat_id),
            "type": resolved.value,
            "enabled": bool(new_state),
            "all_enabled": sorted(t.value for t in full_set),
        }

    def drain_alerts(self, chat_id: int, *, limit: int = 50) -> list[dict[str, Any]]:
        if self._alerts is None:
            return []
        alerts = self._alerts.drain(chat_id, limit=limit)
        return [a.to_dict() for a in alerts]


# Human-readable labels for each alert type. Kept beside the controller
# (not the alerts module) because they're a presentation concern that
# the Telegram service ultimately serves to the user.
_ALERT_LABELS: dict[AlertType, str] = {
    AlertType.TRADE_OPENED:       "Trade opened (BUY)",
    AlertType.TRADE_CLOSED:       "Trade closed (CLOSE / SL / TP)",
    AlertType.TRADE_FAILED:       "Trade failed or ambiguous",
    AlertType.PANIC_CLOSE:        "Panic close completed",
    AlertType.DAILY_LOSS_HALT:    "Daily-loss kill switch fired",
    AlertType.CYCLE_ERROR:        "Cycle crashed",
    AlertType.AI_UNAVAILABLE:     "AI / LLM unavailable",
    AlertType.UNIVERSE_CHANGED:   "Tracked universe changed",
    AlertType.UNIVERSE_REJECTED:  "Universe candidates rejected (activity filter)",
    AlertType.BOT_PAUSED_RESUMED: "Bot paused / resumed",
    AlertType.STRATEGY_AUTOTUNED: "Strategy autotuned by manager LLM",
}
