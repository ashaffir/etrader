"""Tool base types — the contract every tool implements.

A tool reads a :class:`ToolContext` (the per-instrument, per-cycle
state) and produces a :class:`ToolResult`. Two modes exist:

- **feature** tools enrich the candidate with a value the LLM sees in
  the prompt (and the deterministic fallback can score against);
- **gate** tools veto the candidate when a hard precondition fails
  (e.g. spread is too wide, market is closed). A gated candidate
  never reaches the LLM, saving cost.

Some tools play both roles (e.g. ``trend_filter`` exposes its score
as a feature AND vetoes when the higher-TF trend opposes the
candidate). That's allowed by setting ``role = "both"``.

The base class avoids dependency on the rest of the strategy module
so individual tool files can import from ``base`` without pulling in
the registry or runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from ...config import GuardrailsConfig, StrategyConfig
from ...etoro.market_data import Candle, InstrumentMeta, LiveRate


class AssetClass(str, Enum):
    """Coarse classification used by the selector to gate volume/market-hours tools."""

    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    INDEX = "index"
    COMMODITY = "commodity"
    FX = "fx"
    OTHER = "other"


# eToro instrumentTypeID mapping (best-effort; these are stable enough to ship):
# 1 = stock, 4 = ETF, 5 = crypto, 6 = currency/FX, 8 = index, 10 = commodity.
_TYPE_ID_TO_CLASS: dict[int, AssetClass] = {
    1: AssetClass.STOCK,
    4: AssetClass.ETF,
    5: AssetClass.CRYPTO,
    6: AssetClass.FX,
    8: AssetClass.INDEX,
    10: AssetClass.COMMODITY,
}

_KNOWN_CRYPTO = {"BTC", "ETH", "SOL", "ADA", "DOGE", "XRP", "LTC", "MATIC", "DOT", "BNB"}
_KNOWN_INDEX = {"SPX500", "NSDQ100", "DJ30", "RUSS2000", "DAX30", "UK100", "JPN225"}


def asset_class_for(meta: InstrumentMeta | None, *, symbol: str | None = None) -> AssetClass:
    """Return the asset class for an instrument, with safe fallbacks.

    Used by the selector to decide which tools make sense; e.g. FX
    instruments don't have meaningful candle volume on eToro, so
    volume tools are auto-disabled for them.
    """
    if meta is not None and meta.instrument_type_id in _TYPE_ID_TO_CLASS:
        return _TYPE_ID_TO_CLASS[int(meta.instrument_type_id)]
    sym = (symbol or (meta.symbol_full if meta else None) or "").upper()
    if sym in _KNOWN_CRYPTO:
        return AssetClass.CRYPTO
    if sym in _KNOWN_INDEX:
        return AssetClass.INDEX
    return AssetClass.OTHER


# ---------------------------------------------------------------------------
# Context + result types
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Everything a tool might need to evaluate one candidate."""

    instrument_id: int
    symbol: str
    asset_class: AssetClass
    candidate_action: str         # "BUY" | "CLOSE"
    strategy: StrategyConfig
    guardrails: GuardrailsConfig
    candles: Sequence[Candle]
    rate: LiveRate | None
    instrument_meta: InstrumentMeta | None
    higher_tf_candles: Sequence[Candle] = ()    # daily candles, may be empty
    cross_asset_regime: Mapping[str, Any] | None = None
    feed_summary: Mapping[str, Any] | None = None  # filled by feed tool
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_volume(self) -> bool:
        return any((c.volume or 0) > 0 for c in self.candles)

    def closes(self) -> list[float]:
        return [c.close for c in self.candles if c.close > 0]

    def highs(self) -> list[float]:
        return [c.high for c in self.candles]

    def lows(self) -> list[float]:
        return [c.low for c in self.candles]

    def volumes(self) -> list[float]:
        return [c.volume or 0.0 for c in self.candles]


@dataclass(frozen=True)
class ToolResult:
    """A tool's verdict for one candidate.

    - ``features`` — a dict of named values surfaced to the LLM. Values
      should be JSON-serializable (numbers, strings, small lists/dicts).
    - ``score`` — optional +1..-1 directional score. ``+1`` strongly
      supports the candidate's action; ``-1`` strongly opposes; ``0``
      neutral. Used by the deterministic fallback's weighted vote.
    - ``gate_passed`` — for gate-mode tools. ``False`` vetoes the candidate.
    - ``gate_reason`` — human-readable reason on veto.
    """

    features: Mapping[str, Any] = field(default_factory=dict)
    score: float | None = None
    gate_passed: bool = True
    gate_reason: str = ""


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class Tool:
    """Abstract base. Concrete tools subclass and override ``evaluate``.

    Subclasses set the class attributes ``name``, ``family``,
    ``role``, ``asset_classes``, and ``purpose``. The base provides
    defensive ``__init_subclass__`` validation so a typo doesn't
    silently disable a tool.
    """

    name: str = ""                      # unique key; e.g. "sma_cross"
    family: str = "price"               # "price" | "volume" | "context"
    role: str = "feature"               # "feature" | "gate" | "both"
    asset_classes: tuple[AssetClass, ...] = (
        AssetClass.STOCK,
        AssetClass.CRYPTO,
        AssetClass.ETF,
        AssetClass.INDEX,
        AssetClass.COMMODITY,
        AssetClass.FX,
        AssetClass.OTHER,
    )
    purpose: str = ""                   # short description for /signals output
    requires_volume: bool = False       # selector skips when ctx.has_volume is False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name and cls.role not in {"feature", "gate", "both"}:
            raise ValueError(f"Tool {cls.name!r}: role must be feature|gate|both")
        if cls.name and cls.family not in {"price", "volume", "context"}:
            raise ValueError(f"Tool {cls.name!r}: family must be price|volume|context")

    def applies_to(self, ctx: ToolContext) -> bool:
        """Cheap precheck the selector calls before adding the tool to the run set."""
        if ctx.asset_class not in self.asset_classes:
            return False
        if self.requires_volume and not ctx.has_volume:
            return False
        return True

    def evaluate(self, ctx: ToolContext) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__}.evaluate() not implemented")
