"""Build / refresh the set of instruments the bot tracks.

The universe = base symbols (always tracked) + optional LLM-suggested
rotation symbols, capped at ``UniverseConfig.max_tracked``. Symbols are
resolved to instrument IDs once and stored in
:class:`~src.etoro.instrument_cache.InstrumentCache`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..ai.azure_client import AzureFoundryClient, AzureUnavailable
from ..ai.prompts import build_universe_rotation_prompt
from ..config import UniverseConfig
from ..etoro.client import EtoroClient
from ..etoro.errors import EtoroApiError
from ..etoro.instrument_cache import InstrumentCache
from ..etoro.market_data import search_instrument


@dataclass(frozen=True)
class TrackedUniverse:
    instrument_ids: list[int]
    symbol_for_id: dict[int, str]
    base_count: int
    llm_count: int

    def __len__(self) -> int:
        return len(self.instrument_ids)


class UniverseBuilder:
    def __init__(
        self,
        cfg: UniverseConfig,
        *,
        cache: InstrumentCache,
        ai_client: AzureFoundryClient | None,
        etoro_client: EtoroClient,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._cfg = cfg
        self._cache = cache
        self._ai = ai_client
        self._etoro = etoro_client
        self._logger = logger or logging.getLogger("etrader.strategy.universe")

    def build(self, *, market_context: str | None = None) -> TrackedUniverse:
        base_ids: list[int] = []
        base_symbol_for_id: dict[int, str] = {}
        for sym in self._cfg.base_symbols:
            inst_id = self._resolve(sym)
            if inst_id is not None:
                base_ids.append(inst_id)
                base_symbol_for_id[inst_id] = sym.upper()

        room = max(0, self._cfg.max_tracked - len(base_ids))
        llm_ids: list[int] = []
        llm_symbol_for_id: dict[int, str] = {}

        if room > 0 and self._cfg.enable_llm_rotation and self._ai is not None:
            extra_symbols = self._llm_rotation_suggestions(
                exclude=tuple(self._cfg.base_symbols), max_count=room, context=market_context
            )
            for sym in extra_symbols:
                if len(llm_ids) >= room:
                    break
                inst_id = self._resolve(sym)
                if inst_id is None or inst_id in base_symbol_for_id:
                    continue
                llm_ids.append(inst_id)
                llm_symbol_for_id[inst_id] = sym.upper()

        # Persist the cache once after all resolutions
        try:
            self._cache.save()
        except OSError as exc:
            self._logger.warning("failed to save instrument cache: %s", exc)

        all_ids = base_ids + llm_ids
        symbol_for_id = {**base_symbol_for_id, **llm_symbol_for_id}
        return TrackedUniverse(
            instrument_ids=all_ids,
            symbol_for_id=symbol_for_id,
            base_count=len(base_ids),
            llm_count=len(llm_ids),
        )

    # ------------------------------------------------------------------

    def _resolve(self, symbol: str) -> int | None:
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached
        try:
            inst_id = search_instrument(self._etoro, symbol)
        except EtoroApiError as exc:
            self._logger.warning("symbol resolution failed for %s: %s", symbol, exc)
            return None
        if inst_id is None:
            self._logger.warning("symbol %s not found on eToro", symbol)
            return None
        self._cache.upsert(symbol, inst_id)
        return inst_id

    def _llm_rotation_suggestions(
        self,
        *,
        exclude: Sequence[str],
        max_count: int,
        context: str | None,
    ) -> list[str]:
        assert self._ai is not None
        excluded_lower = {s.upper() for s in exclude}
        system, user = build_universe_rotation_prompt(
            base_symbols=exclude,
            excluded_symbols=exclude,
            max_count=max_count,
            market_context=context,
        )
        try:
            result = self._ai.chat_json(system=system, user=user, require_json=True)
        except AzureUnavailable as exc:
            self._logger.warning("LLM rotation unavailable: %s", exc)
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
            if not sym or sym in excluded_lower or sym in seen:
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
    out = dict(universe.symbol_for_id)
    for inst_id, sym in extra.items():
        out.setdefault(inst_id, sym)
    return out


def collect_owned_instrument_ids(positions: Iterable) -> set[int]:
    ids: set[int] = set()
    for p in positions:
        inst_id = getattr(p, "instrument_id", None)
        if inst_id is not None:
            ids.add(int(inst_id))
    return ids
