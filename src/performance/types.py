"""Dataclasses shared by the performance package.

Three records describe the bot's track record:

- :class:`OpenTradeState` — what we know about a bot-owned position
  while it is still open (entry + running mark-to-market with MFE/MAE).
- :class:`RealizedTrade` — written once when the position closes;
  pinned for life in the closed-trade ledger.
- :class:`DailySnapshot` — one row per UTC day with the bot's
  cumulative track record at end-of-day.

Kept tiny and dependency-free on purpose; the storage and tracker
modules import these without pulling in any heavy machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpenTradeState:
    """A bot-owned position the tracker is mark-to-marketing each cycle."""

    position_id: int
    instrument_id: int
    symbol: str
    asset_class: str             # snapshot of AssetClass.value at open
    is_buy: bool
    amount_usd: float
    units: float
    open_rate: float
    opened_at_iso: str           # ISO-8601 UTC
    # Running state — updated on every cycle that sees this position.
    last_mark: float | None = None
    last_pnl_usd: float | None = None
    last_pnl_pct: float | None = None
    last_seen_iso: str | None = None
    # MFE = Max Favourable Excursion (peak P/L while open, positive).
    # MAE = Max Adverse Excursion (worst P/L while open, negative).
    mfe_usd: float = 0.0
    mae_usd: float = 0.0
    snapshots: int = 0           # how many mark-to-markets we've recorded

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "is_buy": self.is_buy,
            "amount_usd": self.amount_usd,
            "units": self.units,
            "open_rate": self.open_rate,
            "opened_at_iso": self.opened_at_iso,
            "last_mark": self.last_mark,
            "last_pnl_usd": self.last_pnl_usd,
            "last_pnl_pct": self.last_pnl_pct,
            "last_seen_iso": self.last_seen_iso,
            "mfe_usd": self.mfe_usd,
            "mae_usd": self.mae_usd,
            "snapshots": self.snapshots,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OpenTradeState":
        return cls(
            position_id=int(d.get("position_id") or 0),
            instrument_id=int(d.get("instrument_id") or 0),
            symbol=str(d.get("symbol") or ""),
            asset_class=str(d.get("asset_class") or "other"),
            is_buy=bool(d.get("is_buy", True)),
            amount_usd=float(d.get("amount_usd") or 0.0),
            units=float(d.get("units") or 0.0),
            open_rate=float(d.get("open_rate") or 0.0),
            opened_at_iso=str(d.get("opened_at_iso") or ""),
            last_mark=_opt_float(d.get("last_mark")),
            last_pnl_usd=_opt_float(d.get("last_pnl_usd")),
            last_pnl_pct=_opt_float(d.get("last_pnl_pct")),
            last_seen_iso=d.get("last_seen_iso"),
            mfe_usd=float(d.get("mfe_usd") or 0.0),
            mae_usd=float(d.get("mae_usd") or 0.0),
            snapshots=int(d.get("snapshots") or 0),
        )


@dataclass(frozen=True)
class RealizedTrade:
    """A bot-owned position after it closes — append-only record."""

    position_id: int
    instrument_id: int
    symbol: str
    asset_class: str
    is_buy: bool
    amount_usd: float
    units: float
    open_rate: float
    close_rate: float | None        # None when we couldn't observe a final mark
    realized_pnl_usd: float
    realized_pnl_pct: float
    opened_at_iso: str
    closed_at_iso: str
    hold_seconds: int
    mfe_usd: float
    mae_usd: float
    close_reason: str               # "reconciled" | "panic" | "external" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "is_buy": self.is_buy,
            "amount_usd": self.amount_usd,
            "units": self.units,
            "open_rate": self.open_rate,
            "close_rate": self.close_rate,
            "realized_pnl_usd": self.realized_pnl_usd,
            "realized_pnl_pct": self.realized_pnl_pct,
            "opened_at_iso": self.opened_at_iso,
            "closed_at_iso": self.closed_at_iso,
            "hold_seconds": self.hold_seconds,
            "mfe_usd": self.mfe_usd,
            "mae_usd": self.mae_usd,
            "close_reason": self.close_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RealizedTrade":
        return cls(
            position_id=int(d.get("position_id") or 0),
            instrument_id=int(d.get("instrument_id") or 0),
            symbol=str(d.get("symbol") or ""),
            asset_class=str(d.get("asset_class") or "other"),
            is_buy=bool(d.get("is_buy", True)),
            amount_usd=float(d.get("amount_usd") or 0.0),
            units=float(d.get("units") or 0.0),
            open_rate=float(d.get("open_rate") or 0.0),
            close_rate=_opt_float(d.get("close_rate")),
            realized_pnl_usd=float(d.get("realized_pnl_usd") or 0.0),
            realized_pnl_pct=float(d.get("realized_pnl_pct") or 0.0),
            opened_at_iso=str(d.get("opened_at_iso") or ""),
            closed_at_iso=str(d.get("closed_at_iso") or ""),
            hold_seconds=int(d.get("hold_seconds") or 0),
            mfe_usd=float(d.get("mfe_usd") or 0.0),
            mae_usd=float(d.get("mae_usd") or 0.0),
            close_reason=str(d.get("close_reason") or "unknown"),
        )


@dataclass
class DailySnapshot:
    """One row per UTC day — written on first cycle of a new day and on shutdown."""

    date_iso: str                                # YYYY-MM-DD UTC
    equity_open: float | None = None             # equity on first cycle of day
    equity_close: float | None = None            # equity on most-recent cycle of day
    equity_high: float | None = None
    equity_low: float | None = None
    bot_unrealized_open_usd: float | None = None
    bot_unrealized_close_usd: float | None = None
    bot_realized_today_usd: float = 0.0
    account_unrealized_close_usd: float | None = None
    bot_trades_today: int = 0
    bot_wins_today: int = 0
    bot_losses_today: int = 0
    bot_breakeven_today: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_iso": self.date_iso,
            "equity_open": self.equity_open,
            "equity_close": self.equity_close,
            "equity_high": self.equity_high,
            "equity_low": self.equity_low,
            "bot_unrealized_open_usd": self.bot_unrealized_open_usd,
            "bot_unrealized_close_usd": self.bot_unrealized_close_usd,
            "bot_realized_today_usd": self.bot_realized_today_usd,
            "account_unrealized_close_usd": self.account_unrealized_close_usd,
            "bot_trades_today": self.bot_trades_today,
            "bot_wins_today": self.bot_wins_today,
            "bot_losses_today": self.bot_losses_today,
            "bot_breakeven_today": self.bot_breakeven_today,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DailySnapshot":
        return cls(
            date_iso=str(d.get("date_iso") or ""),
            equity_open=_opt_float(d.get("equity_open")),
            equity_close=_opt_float(d.get("equity_close")),
            equity_high=_opt_float(d.get("equity_high")),
            equity_low=_opt_float(d.get("equity_low")),
            bot_unrealized_open_usd=_opt_float(d.get("bot_unrealized_open_usd")),
            bot_unrealized_close_usd=_opt_float(d.get("bot_unrealized_close_usd")),
            bot_realized_today_usd=float(d.get("bot_realized_today_usd") or 0.0),
            account_unrealized_close_usd=_opt_float(d.get("account_unrealized_close_usd")),
            bot_trades_today=int(d.get("bot_trades_today") or 0),
            bot_wins_today=int(d.get("bot_wins_today") or 0),
            bot_losses_today=int(d.get("bot_losses_today") or 0),
            bot_breakeven_today=int(d.get("bot_breakeven_today") or 0),
        )


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
