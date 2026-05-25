"""Build / refresh the news-driven tracked-symbol universe.

Phase-2 design (replaces the static ``base_symbols`` flow):

1. **Source candidates** — read from the news pipeline's
   :class:`~src.news.candidate_store.CandidateStore`, ranked by score.
   Optionally augment with operator-supplied seeds (``base_symbols`` —
   normally empty) and an LLM rotation fallback if the LLM is wired
   in and the news pool is too small.
2. **Resolve to instrument IDs** via :class:`~src.etoro.instrument_cache.InstrumentCache`
   plus :func:`~src.etoro.market_data.search_instrument` for cold lookups.
3. **Probe** the top ``probe_max_candidates`` for live rates + candles
   so we can compute spread% and ATR%.
4. **Gate** with :class:`~src.strategy.activity_filter.ActivityFilter`
   — symbols too flat (low ATR) or too expensive (wide spread) are
   rejected with a recorded reason.
5. **Compose** the tracked universe: owned positions always pass
   through (we can't lose sight of them), then survivors fill up to
   ``max_tracked``. If we don't have enough qualified candidates, the
   universe shrinks rather than padding with random symbols.

Every admitted instrument carries a human-readable ``reason`` that
the ``/universe`` and ``/news`` Telegram commands display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from ..ai.azure_client import AzureFoundryClient, AzureUnavailable
from ..ai.prompts import build_universe_rotation_prompt
from ..config import OperationsConfig, UniverseConfig
from ..etoro.client import EtoroClient
from ..etoro.errors import EtoroApiError
from ..etoro.instrument_cache import InstrumentCache
from ..etoro.market_data import (
    Candle,
    LiveRate,
    fetch_candles,
    fetch_rates,
    search_instrument,
)
from ..news.candidate_store import Candidate, CandidateStore
from .activity_filter import ActivityDecision, ActivityFilter


# Source-tag priorities — when survival is competitive, prefer this
# order (lowest = highest priority). Owned positions are always
# admitted; the cap only applies to discretionary slots.
_SOURCE_PRIORITY: Mapping[str, int] = {
    "owned": 0,
    "news": 1,
    "seed": 2,
    "llm": 3,
}


@dataclass(frozen=True)
class TrackedUniverse:
    """Result of one :meth:`UniverseBuilder.build` run."""

    instrument_ids: list[int]
    symbol_for_id: dict[int, str]
    reason_for_id: dict[int, str]
    source_counts: dict[str, int]
    rejected: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instrument_ids)

    @property
    def base_count(self) -> int:
        """Back-compat alias for telemetry: news + seed + owned slots."""
        return (
            self.source_counts.get("news", 0)
            + self.source_counts.get("seed", 0)
            + self.source_counts.get("owned", 0)
        )

    @property
    def llm_count(self) -> int:
        """Back-compat alias for telemetry: LLM-rotation slots."""
        return self.source_counts.get("llm", 0)

    def reason_string(self, instrument_id: int) -> str:
        return self.reason_for_id.get(instrument_id, "")

    def summary_line(self) -> str:
        """Single-line composition summary for logs / Telegram."""
        if not self.source_counts:
            return f"tracked={len(self)} (empty)"
        parts = [f"{k}={v}" for k, v in self.source_counts.items() if v]
        return f"tracked={len(self)} [{', '.join(parts)}]"


# Internal lightweight candidate carrier — used between resolve and
# filter steps. Not exposed.
@dataclass
class _ResolvedCandidate:
    symbol: str
    instrument_id: int
    source: str
    score: float
    reason: str


class UniverseBuilder:
    """News-driven universe builder with an ATR%/spread% activity gate."""

    def __init__(
        self,
        cfg: UniverseConfig,
        operations: OperationsConfig,
        *,
        cache: InstrumentCache,
        candidate_store: CandidateStore,
        activity_filter: ActivityFilter,
        ai_client: AzureFoundryClient | None,
        etoro_client: EtoroClient,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
        is_stopping: Callable[[], bool] | None = None,
    ) -> None:
        self._cfg = cfg
        self._ops = operations
        self._cache = cache
        self._store = candidate_store
        self._filter = activity_filter
        self._ai = ai_client
        self._etoro = etoro_client
        self._log = logger or logging.getLogger("etrader.strategy.universe")
        # Optional callable that returns True when the bot is shutting
        # down. The probe loop polls this to bail out of long sequential
        # HTTP chains so Ctrl-C feels responsive.
        self._is_stopping = is_stopping or (lambda: False)

    # ------------------------------------------------------------------

    def build(
        self,
        *,
        must_include: Mapping[int, str] | None = None,
        market_context: str | None = None,
    ) -> TrackedUniverse:
        """Build the next tracked universe.

        ``must_include`` is ``{instrument_id: symbol}`` for positions
        the bot currently owns; those bypass the activity filter and
        the ``max_tracked`` cap so we never lose sight of them.
        """
        must_include = dict(must_include or {})
        cap = max(1, int(self._cfg.max_tracked))

        owned_resolved: list[_ResolvedCandidate] = [
            _ResolvedCandidate(
                symbol=(sym or "").upper() or f"INST-{inst_id}",
                instrument_id=int(inst_id),
                source="owned",
                score=float("inf"),
                reason="owned position (auto-included)",
            )
            for inst_id, sym in must_include.items()
        ]

        # ---- News candidates ------------------------------------------------
        self._store.prune()
        candidates = self._store.top(self._cfg.probe_max_candidates)
        news_resolved = self._resolve_candidates(candidates, source="news")

        # ---- Seeds (legacy base_symbols) -----------------------------------
        seed_resolved: list[_ResolvedCandidate] = []
        if self._cfg.base_symbols:
            self._log.warning(
                "[universe] %d seed symbol(s) configured via base_symbols — "
                "they still pass through the activity filter.",
                len(self._cfg.base_symbols),
            )
            seed_resolved = self._resolve_seed_symbols(self._cfg.base_symbols)

        # ---- Activity-filter pass ------------------------------------------
        # Owned positions skip the filter (we must keep tracking them).
        to_probe = self._dedupe_by_id(news_resolved + seed_resolved, exclude=owned_resolved)
        probe_data = self._probe(to_probe)
        accepted, rejected = self._apply_filter(to_probe, probe_data, cap=cap - len(owned_resolved))

        # ---- LLM fallback (only if we're still short on capacity) ----------
        llm_resolved: list[_ResolvedCandidate] = []
        remaining = cap - len(owned_resolved) - len(accepted)
        if (
            remaining > 0
            and self._cfg.enable_llm_rotation
            and self._ai is not None
        ):
            excluded = {c.symbol for c in owned_resolved + accepted}
            extra_syms = self._llm_rotation_suggestions(
                excluded=excluded, max_count=remaining, context=market_context,
            )
            llm_resolved = self._resolve_symbols(extra_syms, source="llm",
                                                 reason_prefix="LLM rotation")
            llm_to_probe = self._dedupe_by_id(
                llm_resolved, exclude=owned_resolved + accepted,
            )
            llm_probe = self._probe(llm_to_probe)
            llm_accepted, llm_rejected = self._apply_filter(
                llm_to_probe, llm_probe, cap=remaining,
            )
            accepted.extend(llm_accepted)
            rejected.update(llm_rejected)

        # ---- Compose final universe ----------------------------------------
        try:
            self._cache.save()
        except OSError as exc:
            self._log.warning("[universe] failed to save instrument cache: %s", exc)

        all_accepted = owned_resolved + accepted
        instrument_ids = [c.instrument_id for c in all_accepted]
        symbol_for_id = {c.instrument_id: c.symbol for c in all_accepted}
        reason_for_id = {c.instrument_id: c.reason for c in all_accepted}
        source_counts: dict[str, int] = {}
        for c in all_accepted:
            source_counts[c.source] = source_counts.get(c.source, 0) + 1

        universe = TrackedUniverse(
            instrument_ids=instrument_ids,
            symbol_for_id=symbol_for_id,
            reason_for_id=reason_for_id,
            source_counts=source_counts,
            rejected=rejected,
        )
        self._log.info("[universe] composed → %s", universe.summary_line())
        if rejected:
            sample = ", ".join(f"{s}: {r}" for s, r in list(rejected.items())[:5])
            self._log.info("[universe] %d rejected (sample: %s)", len(rejected), sample)
        if not universe.instrument_ids:
            self._log.warning(
                "[universe] EMPTY — no candidate passed the activity filter and "
                "no owned positions. Bot will idle until the next news scan.",
            )
        return universe

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve_candidates(
        self,
        candidates: Sequence[Candidate],
        *,
        source: str,
    ) -> list[_ResolvedCandidate]:
        out: list[_ResolvedCandidate] = []
        for cand in candidates:
            inst_id = self._resolve_symbol(cand.symbol)
            if inst_id is None:
                continue
            reason = cand.reason
            out.append(
                _ResolvedCandidate(
                    symbol=cand.symbol.upper(),
                    instrument_id=inst_id,
                    source=source,
                    score=cand.score,
                    reason=reason,
                )
            )
        return out

    def _resolve_seed_symbols(
        self, symbols: Sequence[str]
    ) -> list[_ResolvedCandidate]:
        out: list[_ResolvedCandidate] = []
        for sym in symbols:
            inst_id = self._resolve_symbol(sym)
            if inst_id is None:
                continue
            out.append(
                _ResolvedCandidate(
                    symbol=sym.upper(),
                    instrument_id=inst_id,
                    source="seed",
                    score=0.0,
                    reason="seed: configured in [universe].base_symbols",
                )
            )
        return out

    def _resolve_symbols(
        self,
        symbols: Iterable[str],
        *,
        source: str,
        reason_prefix: str,
    ) -> list[_ResolvedCandidate]:
        out: list[_ResolvedCandidate] = []
        for sym in symbols:
            sym = sym.strip().upper()
            if not sym:
                continue
            inst_id = self._resolve_symbol(sym)
            if inst_id is None:
                continue
            out.append(
                _ResolvedCandidate(
                    symbol=sym,
                    instrument_id=inst_id,
                    source=source,
                    score=0.0,
                    reason=f"{reason_prefix}: {sym}",
                )
            )
        return out

    def _resolve_symbol(self, symbol: str) -> int | None:
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached
        try:
            inst_id = search_instrument(self._etoro, symbol)
        except EtoroApiError as exc:
            self._log.warning("[universe] symbol resolution failed for %s: %s", symbol, exc)
            return None
        if inst_id is None:
            self._log.info("[universe] symbol %s not found on eToro", symbol)
            return None
        self._cache.upsert(symbol, inst_id)
        return inst_id

    # ------------------------------------------------------------------
    # Probing + filtering
    # ------------------------------------------------------------------

    def _dedupe_by_id(
        self,
        items: Sequence[_ResolvedCandidate],
        *,
        exclude: Sequence[_ResolvedCandidate] = (),
    ) -> list[_ResolvedCandidate]:
        already = {c.instrument_id for c in exclude}
        seen: set[int] = set()
        out: list[_ResolvedCandidate] = []
        for c in items:
            if c.instrument_id in already or c.instrument_id in seen:
                continue
            seen.add(c.instrument_id)
            out.append(c)
        return out

    def _probe(
        self, candidates: Sequence[_ResolvedCandidate]
    ) -> tuple[dict[int, LiveRate], dict[int, list[Candle]]]:
        """Fetch rates + candles for the supplied candidates.

        Each call is wrapped so a single failure on one instrument
        doesn't drop the whole batch. We don't retry beyond what
        eToro's client wrapper already does.
        """
        if not candidates:
            return {}, {}
        ids = [c.instrument_id for c in candidates]
        rates: dict[int, LiveRate] = {}
        try:
            rates = fetch_rates(self._etoro, ids)
        except EtoroApiError as exc:
            self._log.warning("[universe] probe rates failed: %s", exc)

        candles: dict[int, list[Candle]] = {}
        for cand in candidates:
            # The probe makes one HTTP call per candidate. With up to
            # ``probe_max_candidates`` (default 50) that's a long chain;
            # poll the stop callable so Ctrl-C aborts in <1 candle.
            if self._is_stopping():
                self._log.info(
                    "[universe] stop requested mid-probe — keeping %d/%d candles",
                    len(candles), len(candidates),
                )
                break
            try:
                candles[cand.instrument_id] = fetch_candles(
                    self._etoro,
                    cand.instrument_id,
                    interval=self._ops.candle_interval,
                    count=self._ops.candle_count,
                )
            except EtoroApiError as exc:
                self._log.warning(
                    "[universe] probe candles failed for %s: %s", cand.symbol, exc,
                )
                candles[cand.instrument_id] = []
        return rates, candles

    def _apply_filter(
        self,
        candidates: Sequence[_ResolvedCandidate],
        probe: tuple[dict[int, LiveRate], dict[int, list[Candle]]],
        *,
        cap: int,
    ) -> tuple[list[_ResolvedCandidate], dict[str, str]]:
        """Run the activity filter; respect the supplied admission cap."""
        rates, candles = probe
        accepted: list[_ResolvedCandidate] = []
        rejected: dict[str, str] = {}
        if cap <= 0:
            for c in candidates:
                rejected[c.symbol] = "skipped: capacity full"
            return accepted, rejected

        # Sort by source priority (lowest first) then by descending score
        # so higher-priority sources fill the budget first.
        ordered = sorted(
            candidates,
            key=lambda c: (_SOURCE_PRIORITY.get(c.source, 99), -c.score, c.symbol),
        )
        for cand in ordered:
            if len(accepted) >= cap:
                rejected[cand.symbol] = "deferred: capacity full"
                continue
            decision: ActivityDecision = self._filter.evaluate(
                candles=candles.get(cand.instrument_id, []),
                rate=rates.get(cand.instrument_id),
            )
            if decision.passed:
                cand_with_metrics = _ResolvedCandidate(
                    symbol=cand.symbol,
                    instrument_id=cand.instrument_id,
                    source=cand.source,
                    score=cand.score,
                    reason=f"{cand.reason} | {decision.short_summary()}",
                )
                accepted.append(cand_with_metrics)
            else:
                rejected[cand.symbol] = decision.reason
        return accepted, rejected

    # ------------------------------------------------------------------
    # LLM rotation (fallback only)
    # ------------------------------------------------------------------

    def _llm_rotation_suggestions(
        self,
        *,
        excluded: Iterable[str],
        max_count: int,
        context: str | None,
    ) -> list[str]:
        assert self._ai is not None
        excluded_upper = {s.upper() for s in excluded}
        system, user = build_universe_rotation_prompt(
            base_symbols=tuple(excluded_upper),
            excluded_symbols=tuple(excluded_upper),
            max_count=max_count,
            market_context=context,
        )
        try:
            result = self._ai.chat_json(system=system, user=user, require_json=True)
        except AzureUnavailable as exc:
            self._log.warning("[universe] LLM rotation unavailable: %s", exc)
            return []
        if not isinstance(result.parsed_json, dict):
            return []
        symbols = result.parsed_json.get("symbols")
        if not isinstance(symbols, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for s in symbols:
            if not isinstance(s, str):
                continue
            sym = s.strip().upper()
            if not sym or sym in excluded_upper or sym in seen:
                continue
            seen.add(sym)
            cleaned.append(sym)
            if len(cleaned) >= max_count:
                break
        return cleaned


def merge_known_symbols(
    universe: TrackedUniverse,
    extra: Mapping[int, str],
) -> Mapping[int, str]:
    """Back-compat helper used by ``cycle.py`` callers.

    Combines a universe's ``symbol_for_id`` with caller-supplied
    extras, preferring the universe's mapping for collisions.
    """
    out = dict(universe.symbol_for_id)
    for inst_id, sym in extra.items():
        out.setdefault(inst_id, sym)
    return out


def collect_owned_instrument_ids(positions: Iterable) -> set[int]:
    """Extract `instrument_id` values from an iterable of positions."""
    ids: set[int] = set()
    for p in positions:
        inst_id = getattr(p, "instrument_id", None)
        if inst_id is not None:
            ids.add(int(inst_id))
    return ids
