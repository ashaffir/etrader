"""SQL constants + row ↔ dataclass converters for ``PerformanceStorage``.

Kept private (leading underscore) and column-positional so a column
re-order in the schema can't silently corrupt persisted data — every
INSERT is matched by an explicit positional helper.
"""

from __future__ import annotations

import sqlite3

from .types import DailySnapshot, OpenTradeState, RealizedTrade


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS open_positions (
    position_id     INTEGER PRIMARY KEY,
    instrument_id   INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    asset_class     TEXT NOT NULL,
    is_buy          INTEGER NOT NULL,
    amount_usd      REAL NOT NULL,
    units           REAL NOT NULL,
    open_rate       REAL NOT NULL,
    opened_at_iso   TEXT NOT NULL,
    last_mark       REAL,
    last_pnl_usd    REAL,
    last_pnl_pct    REAL,
    last_seen_iso   TEXT,
    mfe_usd         REAL NOT NULL DEFAULT 0,
    mae_usd         REAL NOT NULL DEFAULT 0,
    snapshots       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS closed_trades (
    position_id      INTEGER PRIMARY KEY,
    instrument_id    INTEGER NOT NULL,
    symbol           TEXT NOT NULL,
    asset_class      TEXT NOT NULL,
    is_buy           INTEGER NOT NULL,
    amount_usd       REAL NOT NULL,
    units            REAL NOT NULL,
    open_rate        REAL NOT NULL,
    close_rate       REAL,
    realized_pnl_usd REAL NOT NULL,
    realized_pnl_pct REAL NOT NULL,
    opened_at_iso    TEXT NOT NULL,
    closed_at_iso    TEXT NOT NULL,
    hold_seconds     INTEGER NOT NULL,
    mfe_usd          REAL NOT NULL,
    mae_usd          REAL NOT NULL,
    close_reason     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_closed_trades_closed_at
    ON closed_trades(closed_at_iso);
CREATE INDEX IF NOT EXISTS idx_closed_trades_symbol
    ON closed_trades(symbol);

CREATE TABLE IF NOT EXISTS daily_snapshots (
    date_iso                     TEXT PRIMARY KEY,
    equity_open                  REAL,
    equity_close                 REAL,
    equity_high                  REAL,
    equity_low                   REAL,
    bot_unrealized_open_usd      REAL,
    bot_unrealized_close_usd     REAL,
    bot_realized_today_usd       REAL NOT NULL DEFAULT 0,
    account_unrealized_close_usd REAL,
    bot_trades_today             INTEGER NOT NULL DEFAULT 0,
    bot_wins_today               INTEGER NOT NULL DEFAULT 0,
    bot_losses_today             INTEGER NOT NULL DEFAULT 0,
    bot_breakeven_today          INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Prepared statements
# ---------------------------------------------------------------------------

INSERT_OPEN = """
INSERT INTO open_positions (
    position_id, instrument_id, symbol, asset_class, is_buy,
    amount_usd, units, open_rate, opened_at_iso,
    last_mark, last_pnl_usd, last_pnl_pct, last_seen_iso,
    mfe_usd, mae_usd, snapshots
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_OR_REPLACE_CLOSED = """
INSERT OR REPLACE INTO closed_trades (
    position_id, instrument_id, symbol, asset_class, is_buy,
    amount_usd, units, open_rate, close_rate,
    realized_pnl_usd, realized_pnl_pct,
    opened_at_iso, closed_at_iso, hold_seconds,
    mfe_usd, mae_usd, close_reason
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPSERT_DAILY = """
INSERT INTO daily_snapshots (
    date_iso, equity_open, equity_close, equity_high, equity_low,
    bot_unrealized_open_usd, bot_unrealized_close_usd,
    bot_realized_today_usd, account_unrealized_close_usd,
    bot_trades_today, bot_wins_today, bot_losses_today, bot_breakeven_today
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(date_iso) DO UPDATE SET
    equity_open                  = excluded.equity_open,
    equity_close                 = excluded.equity_close,
    equity_high                  = excluded.equity_high,
    equity_low                   = excluded.equity_low,
    bot_unrealized_open_usd      = excluded.bot_unrealized_open_usd,
    bot_unrealized_close_usd     = excluded.bot_unrealized_close_usd,
    bot_realized_today_usd       = excluded.bot_realized_today_usd,
    account_unrealized_close_usd = excluded.account_unrealized_close_usd,
    bot_trades_today             = excluded.bot_trades_today,
    bot_wins_today               = excluded.bot_wins_today,
    bot_losses_today             = excluded.bot_losses_today,
    bot_breakeven_today          = excluded.bot_breakeven_today
"""


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------

def open_state_to_row(s: OpenTradeState) -> tuple:
    return (
        int(s.position_id),
        int(s.instrument_id),
        str(s.symbol or ""),
        str(s.asset_class or "other"),
        1 if s.is_buy else 0,
        float(s.amount_usd or 0.0),
        float(s.units or 0.0),
        float(s.open_rate or 0.0),
        str(s.opened_at_iso or ""),
        s.last_mark,
        s.last_pnl_usd,
        s.last_pnl_pct,
        s.last_seen_iso,
        float(s.mfe_usd or 0.0),
        float(s.mae_usd or 0.0),
        int(s.snapshots or 0),
    )


def row_to_open_state(row: sqlite3.Row) -> OpenTradeState:
    return OpenTradeState(
        position_id=int(row["position_id"]),
        instrument_id=int(row["instrument_id"]),
        symbol=str(row["symbol"]),
        asset_class=str(row["asset_class"]),
        is_buy=bool(row["is_buy"]),
        amount_usd=float(row["amount_usd"]),
        units=float(row["units"]),
        open_rate=float(row["open_rate"]),
        opened_at_iso=str(row["opened_at_iso"]),
        last_mark=row["last_mark"],
        last_pnl_usd=row["last_pnl_usd"],
        last_pnl_pct=row["last_pnl_pct"],
        last_seen_iso=row["last_seen_iso"],
        mfe_usd=float(row["mfe_usd"]),
        mae_usd=float(row["mae_usd"]),
        snapshots=int(row["snapshots"]),
    )


def closed_trade_to_row(t: RealizedTrade) -> tuple:
    return (
        int(t.position_id),
        int(t.instrument_id),
        str(t.symbol or ""),
        str(t.asset_class or "other"),
        1 if t.is_buy else 0,
        float(t.amount_usd or 0.0),
        float(t.units or 0.0),
        float(t.open_rate or 0.0),
        t.close_rate,
        float(t.realized_pnl_usd or 0.0),
        float(t.realized_pnl_pct or 0.0),
        str(t.opened_at_iso or ""),
        str(t.closed_at_iso or ""),
        int(t.hold_seconds or 0),
        float(t.mfe_usd or 0.0),
        float(t.mae_usd or 0.0),
        str(t.close_reason or "reconciled"),
    )


def row_to_closed_trade(row: sqlite3.Row) -> RealizedTrade:
    return RealizedTrade(
        position_id=int(row["position_id"]),
        instrument_id=int(row["instrument_id"]),
        symbol=str(row["symbol"]),
        asset_class=str(row["asset_class"]),
        is_buy=bool(row["is_buy"]),
        amount_usd=float(row["amount_usd"]),
        units=float(row["units"]),
        open_rate=float(row["open_rate"]),
        close_rate=row["close_rate"],
        realized_pnl_usd=float(row["realized_pnl_usd"]),
        realized_pnl_pct=float(row["realized_pnl_pct"]),
        opened_at_iso=str(row["opened_at_iso"]),
        closed_at_iso=str(row["closed_at_iso"]),
        hold_seconds=int(row["hold_seconds"]),
        mfe_usd=float(row["mfe_usd"]),
        mae_usd=float(row["mae_usd"]),
        close_reason=str(row["close_reason"]),
    )


def daily_to_row(s: DailySnapshot) -> tuple:
    return (
        str(s.date_iso),
        s.equity_open,
        s.equity_close,
        s.equity_high,
        s.equity_low,
        s.bot_unrealized_open_usd,
        s.bot_unrealized_close_usd,
        float(s.bot_realized_today_usd or 0.0),
        s.account_unrealized_close_usd,
        int(s.bot_trades_today or 0),
        int(s.bot_wins_today or 0),
        int(s.bot_losses_today or 0),
        int(s.bot_breakeven_today or 0),
    )


def row_to_daily(row: sqlite3.Row) -> DailySnapshot:
    return DailySnapshot(
        date_iso=str(row["date_iso"]),
        equity_open=row["equity_open"],
        equity_close=row["equity_close"],
        equity_high=row["equity_high"],
        equity_low=row["equity_low"],
        bot_unrealized_open_usd=row["bot_unrealized_open_usd"],
        bot_unrealized_close_usd=row["bot_unrealized_close_usd"],
        bot_realized_today_usd=float(row["bot_realized_today_usd"]),
        account_unrealized_close_usd=row["account_unrealized_close_usd"],
        bot_trades_today=int(row["bot_trades_today"]),
        bot_wins_today=int(row["bot_wins_today"]),
        bot_losses_today=int(row["bot_losses_today"]),
        bot_breakeven_today=int(row["bot_breakeven_today"]),
    )
