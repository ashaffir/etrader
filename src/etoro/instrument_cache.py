"""Persistent symbol ↔ instrument-ID cache.

Serialized to a single JSON file under ``data/instrument_cache.json``.
This keeps repeated runs from re-resolving the same symbols and avoids
hammering the search endpoint on startup.

The cache is **monotonic**: entries are added but never removed. eToro
IDs are stable, so stale entries are harmless. If you ever need to
invalidate, just delete the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass
class InstrumentCache:
    """Bidirectional symbol ↔ id map with disk persistence."""

    path: Path
    symbol_to_id: dict[str, int]
    id_to_symbol: dict[int, str]

    @classmethod
    def load(cls, path: Path) -> "InstrumentCache":
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                raw = {}
        else:
            raw = {}
        symbol_to_id: dict[str, int] = {}
        id_to_symbol: dict[int, str] = {}
        for sym, inst_id in (raw.get("symbol_to_id") or {}).items():
            try:
                inst_id_int = int(inst_id)
            except (TypeError, ValueError):
                continue
            sym_upper = sym.upper()
            symbol_to_id[sym_upper] = inst_id_int
            id_to_symbol[inst_id_int] = sym_upper
        return cls(path=path, symbol_to_id=symbol_to_id, id_to_symbol=id_to_symbol)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"symbol_to_id": {k: v for k, v in self.symbol_to_id.items()}}
        self.path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, symbol: str) -> int | None:
        return self.symbol_to_id.get(symbol.upper())

    def reverse(self, instrument_id: int) -> str | None:
        return self.id_to_symbol.get(int(instrument_id))

    def upsert(self, symbol: str, instrument_id: int) -> None:
        sym = symbol.upper()
        self.symbol_to_id[sym] = int(instrument_id)
        self.id_to_symbol[int(instrument_id)] = sym

    def upsert_many(self, mapping: Mapping[str, int]) -> None:
        for sym, inst_id in mapping.items():
            self.upsert(sym, inst_id)

    def known_symbols(self) -> list[str]:
        return sorted(self.symbol_to_id.keys())
