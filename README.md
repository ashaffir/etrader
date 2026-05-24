# etrader — autonomous eToro trading bot

A Python trading bot that uses the **eToro Public API** for market data and
order execution and an **Azure AI Foundry / Azure OpenAI** deployment as a
decision overlay on top of deterministic technical signals.

The bot:

1. Loads tracked instruments (a curated baseline + optional LLM-suggested
   rotations) into a refreshable universe.
2. Pulls live prices and OHLCV candles every cycle.
3. Runs a **weighted ensemble** of price-action indicators (SMA cross,
   EMA cross, RSI, MACD, Bollinger, Donchian, momentum) and produces a
   shortlist of BUY / CLOSE candidates whenever the combined score
   crosses the configurable entry / exit thresholds.
4. Runs an extensible **tool catalog** (~18 tools across price / volume /
   context families) against each candidate. A regime-aware selector picks
   the relevant subset per (instrument, cycle); hard gates like
   `spread_filter` and `market_hours` can veto a candidate before any LLM
   call. See `/signals` from Telegram for the live rule set.
5. Sends the surviving shortlist plus full portfolio state and the tool
   features to the LLM, which returns a structured JSON action plan.
6. Applies guardrails (cap, parallel limit, cooldown, daily-loss stop, paper
   gate) and either executes or simulates the trades.
7. Verifies trades by re-reading the portfolio after the eToro 10s cache.
8. Persists state (`data/bot_state.json`), pushes alerts, and snapshots
   telemetry that the Telegram service reads through the internal HTTP
   control API.

Two processes:

- **`python -m src.main`** — the trading bot (cycle loop + control HTTP server).
- **`python -m src.telegram_service`** — the Telegram bot (long-polling +
  alert drain + inline-keyboard menus). Optional; the trading bot runs
  fine without it.

Everything is logged to stdout (colored) and to a rotating file.

## Layout

```
etrader/
├── .env                       # eToro + Azure + Telegram credentials (gitignored)
├── .gitignore                 # ignores .env, data/*, logs/*, .venv, caches, IDE
├── config.toml                # behaviour defaults — guardrails, ops, universe, strategy, tools, AI, control, alerting
├── requirements.txt           # runtime deps: requests, openai
├── data/                      # bot-managed runtime state (gitignored)
│   ├── bot_state.json         # cooldowns, owned positions, daily-loss baseline, paused flag
│   ├── config.sqlite          # persisted overrides for any [guardrails]/etc. edited at runtime
│   ├── instrument_cache.json  # symbol → instrumentID resolution cache
│   ├── trade_history.jsonl    # append-only execution log (read by /history)
│   ├── tool_performance.jsonl # rolling per-tool hit-rate stats
│   └── alert_subscriptions.json # per-chat /alerts toggles
├── logs/                      # rotating trader.log
├── src/
│   ├── main.py                # trading bot entry: `python -m src.main`
│   ├── config.py              # .env + TOML + SQLite override loader, schema validation
│   ├── config_store/          # SQLite-backed runtime override store
│   ├── logging_setup.py       # colored stdout + rotating file logger
│   ├── state.py               # in-memory bot state (cooldowns, baseline, owned IDs)
│   ├── persistence.py         # save/load BotState to data/bot_state.json
│   ├── trade_history.py       # append-only data/trade_history.jsonl
│   ├── telemetry.py           # in-memory snapshot store (read by Telegram)
│   ├── alerts.py              # AlertHub + per-chat AlertSubscriptions (Telegram /alerts feed)
│   ├── cycle.py               # one full cycle: fetch → score → decide → risk-gate → execute
│   ├── etoro/                 # API client + endpoint wrappers
│   ├── ai/                    # Azure Foundry chat client + prompts (incl. Q&A)
│   ├── strategy/              # indicators, signals, decisions, universe, risk
│   │   ├── ensemble.py        # weighted score combiner across price tools
│   │   ├── signals.py         # turns indicator scores into BUY / CLOSE candidates
│   │   ├── decisions.py       # LLM (or deterministic) decision engine
│   │   ├── tools/             # extensible tool catalog + selector + perf log
│   │   │   ├── registry.py
│   │   │   ├── selector.py    # picks tools per (instrument, cycle)
│   │   │   ├── price/         # SMA / RSI / MACD / Bollinger / Donchian / momentum
│   │   │   ├── volume_tools.py
│   │   │   └── context_tools.py # market_hours, spread_filter, regime, higher-TF
│   │   ├── tool_orchestration.py
│   │   ├── performance.py     # rolling hit-rate tracker
│   │   ├── regime.py          # cross-asset trending/ranging classifier
│   │   ├── risk.py            # guardrails + daily-loss kill switch
│   │   └── universe.py        # base + LLM-rotated tracked instrument set
│   ├── execution/             # executor (paper/live) + position monitor
│   ├── control/               # internal HTTP control API (consumed by Telegram)
│   │   ├── controller.py      # thread-safe pause/resume/panic/ask/alerts facade
│   │   ├── server.py          # stdlib HTTP server with bearer-token auth
│   │   └── handlers.py        # JSON endpoint dispatch table
│   └── telegram_service/      # SEPARATE PROCESS: Telegram bot poller + dispatcher
│       ├── __main__.py        # entry: `python -m src.telegram_service`
│       ├── bot.py             # long-polling loop + alert drain + callback dispatch
│       ├── commands.py        # parse + dispatch /commands and free-text
│       ├── alerts_menu.py     # /alerts inline-keyboard rendering + callback parsing
│       ├── control_client.py  # requests-based HTTP client for src.control
│       ├── telegram_api.py    # raw Bot API: getUpdates, sendMessage, callback queries
│       └── formatters.py      # render JSON responses as Telegram text
└── tests/                     # unit tests (stdlib unittest, 256 cases)
```

## Setup

```bash
# 1. Make sure .env is populated (PUBLIC_KEY, PRIVATE_KEY, AZURE_*,
#    plus TELEGRAM_BOT_TOKEN / TELEGRAM_ALLOWED_CHAT_IDS /
#    INTERNAL_API_TOKEN if you want the Telegram surface).
# 2. Create a venv and install deps.
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 3. Run the trading bot
python -m src.main

# 4. (optional, separate shell) Run the Telegram service
python -m src.telegram_service
```

See *Tests* below for running the unit-test suite.

## Mode

`config.toml`'s `[mode] trading` toggles paper vs. live:

- `paper` (default) → demo environment, fake balance, real prices. Safe.
- `live` → real environment. Requires `ALLOW_REAL=true` AND `REAL_USER_KEY`
  in `.env`. The bot refuses to start otherwise.

## Guardrails

| Limit                              | Default | Where to change                |
|------------------------------------|---------|--------------------------------|
| Max cash per trade                 | $500    | `config.toml` `[guardrails]`   |
| Max parallel positions (bot-owned) | 10      | same                           |
| Daily-loss kill switch             | $250    | same                           |
| Per-instrument cooldown            | 60 min  | same                           |
| Default stop-loss                  | 5%      | same                           |
| Default take-profit                | 8%      | same                           |
| Max leverage                       | 1       | same                           |

Guardrails are also **editable at runtime** via Telegram (`/set <key>
<value>`). Edits go to a tiny SQLite override store
(`data/config.sqlite`) so they survive restarts — see the
*Configuration* section below.

## Configuration

The bot resolves configuration in three layers (highest precedence
wins):

1. **Dataclass field defaults** — what you get if no other source
   defines a value.
2. **`config.toml`** — bootstrap defaults you ship with the repo.
3. **`data/config.sqlite`** — runtime overrides written by Telegram
   `/set` calls.

On first run the merged defaults are snapshotted into the SQLite
store, so subsequent restarts are DB-authoritative even if someone
edits `config.toml` later. To reset a section to TOML defaults, delete
its rows from `data/config.sqlite` (or just `rm data/config.sqlite` to
re-snapshot from scratch).

Sections persisted in the override DB: `[guardrails]`, `[operations]`,
`[universe]`, `[strategy]`, `[ai]`, `[tools]`, `[logging]`. Secrets and
the `[control]` / `[alerting]` plumbing remain `.env` / TOML-only.

## Strategy

The candidate-selection layer is a **weighted ensemble** over price
indicators. Each enabled indicator emits a signed score in `[-1, +1]`;
the ensemble's `raw_score` is the weight-normalized sum:

```
raw_score = Σ(weight_i * score_i) / Σ(|weight_i|)

raw_score >=  min_signal_strength  → BUY candidate (unowned instrument)
raw_score <= -min_exit_strength    → CLOSE candidate (bot-owned)
```

Defaults: `min_signal_strength=0.40`, `min_exit_strength=0.25` —
exits trip on a weaker signal than entries so the bot cuts losers
fast. Each component (SMA cross, EMA cross, RSI, MACD, Bollinger,
Donchian, momentum) has its own weight in `[strategy]`; set a weight
to `0.0` to disable that indicator's vote without removing it from
the broader tool catalog.

Surviving candidates are then passed through the **tool catalog** in
`src/strategy/tools/`. Tools run in three families:

- **price** — SMA / RSI / MACD / Bollinger / Donchian / momentum (~6 tools)
- **volume** — volume-weighted features (~3 tools)
- **context** — market hours, spread filter, regime, higher-timeframe
  alignment, news feed (~6 tools)

A selector picks an asset-class-compatible subset per (instrument,
cycle) using rolling per-tool hit-rate. Hard gates like
`market_hours` and `spread_filter` can veto a candidate **before any
LLM call** so we don't burn tokens on unactionable trades.

The full live rule set, tool list, and rolling performance is
introspectable from Telegram via `/signals` (alias `/rules`).

## Telegram control surface

The Telegram service is a **separate process** that talks to the trading
bot via an internal HTTP control API (default `http://127.0.0.1:8770`).
Both processes read the same `.env`.

### Setup

1. Get a bot token from `@BotFather` and put it in `.env`:

   ```
   TELEGRAM_BOT_TOKEN=...
   ```

2. DM the bot once with `/start`. Watch the trading bot's logs for the
   line `rejected message from chat_id=<N>` and add that ID to:

   ```
   TELEGRAM_ALLOWED_CHAT_IDS=<N>
   ```

3. Generate a shared secret and put it in `.env`:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
   ```
   INTERNAL_API_TOKEN=...
   ```

4. Run **both** processes (in separate shells):

   ```bash
   . .venv/bin/activate

   # shell 1: trading bot (also boots the control HTTP server)
   python -m src.main

   # shell 2: Telegram service
   python -m src.telegram_service
   ```

### Commands

| Command                       | What it does                                        |
| ----------------------------- | --------------------------------------------------- |
| `/status`                     | Cycle counter, paused?, mode, kill switch, last err |
| `/portfolio`                  | Equity / available / invested / P&L + open posns    |
| `/universe`                   | Instruments currently tracked                       |
| `/signals` (`/rules`)         | Live entry/exit rules + tool catalog + perf stats   |
| `/history [N]`                | Last N trade outcomes (default 20, max 100)         |
| `/guardrails`                 | Show all current `[guardrails]` values              |
| `/set <key> <value>`          | Edit a guardrails field at runtime (mutable)        |
| `/start`, `/resume`           | Resume the trading loop                             |
| `/stop`, `/pause`             | Pause it (open positions are kept)                  |
| `/panic`                      | Close **every** open position (incl. manual) + pause |
| `/panic_bot_only`             | Close only positions opened by this bot + pause     |
| `/ask <question>`             | LLM Q&A with full bot state as context              |
| `/alerts`                     | Open inline-keyboard menu to toggle Telegram alerts |
| Any non-command text          | Treated as `/ask <text>`                            |
| `/help`                       | Print this list                                     |

### Alerts

Tap `/alerts` to open a per-chat submenu of toggleable push-style
alerts. Each row flips ON/OFF on tap, and your choices are persisted in
`data/alert_subscriptions.json`.

By default new chats get the **safety-only** set: `panic_close`,
`daily_loss_halt`, `cycle_error`, `trade_failed`. Opt in to the chatty
trade-by-trade alerts as needed.

Available alert types:

| Type                   | Fires when                                          |
| ---------------------- | --------------------------------------------------- |
| `trade_opened`         | The bot successfully opens a new BUY                |
| `trade_closed`         | The bot closes a position (incl. SL / TP hits)      |
| `trade_failed`         | A trade attempt fails / is ambiguous / rate-limited |
| `panic_close`          | `/panic` (or `/panic_bot_only`) finishes            |
| `daily_loss_halt`      | The daily-loss kill switch fires (edge-triggered)   |
| `cycle_error`          | A cycle raised an uncaught exception                |
| `ai_unavailable`       | LLM availability flips (edge-triggered, both ways)  |
| `universe_changed`     | LLM rotation adds/removes instruments               |
| `bot_paused_resumed`   | `/pause` or `/resume` is invoked                    |

Alerts are queued on the trading bot and drained by the Telegram
service on each poll tick, so they survive short Telegram outages but
are dropped if the queue exceeds `[alerting] max_queue_per_chat` (200
by default — set in `config.toml`).

### Persistence

The trading bot saves its state to `data/bot_state.json` every cycle and
on every `/pause`. On a hard restart it reloads:

- which positions it owns (so `/panic_bot_only` still knows what's the bot's),
- per-instrument cooldowns (re-projected onto the new monotonic clock),
- the daily-loss baseline + halt flag,
- whether the bot was paused.

Trade outcomes are appended to `data/trade_history.jsonl` for the
`/history` command. Alert subscriptions per chat live in
`data/alert_subscriptions.json` (created on first `/alerts` open).

## Activity printout

Every cycle prints a heartbeat block:

```
─── cycle 14 — 2026-05-24T11:36:00Z ───
[universe]  tracking 12 instrument(s): base=10, llm=2
[market]    fetched 12/12 rate(s)
[signals]   3 candidate(s): BUY NVDA(0.62), BUY MSFT(0.48), CLOSE TSLA(-0.31)
[regime]    SPX500 trending up, BTC ranging
[tools]     1 gated: AAPL(spread_filter)
[ai]        decision (llm, 1432 ms): BUY NVDA 250 USD; HOLD others
[risk]      approved 1 / 1 — all clear
[exec]      OK         BUY    NVDA      — orderID 13902598
[portfolio] equity=$10,251.83  available=$9,500.83  invested=$751.00  pnl=$+12.40  bot_owned=4
─────────────────────────────────────
```

DEBUG level adds raw JSON envelopes per call. Per-cycle telemetry is
also pushed to the in-memory snapshot store so `/status`,
`/portfolio`, and `/universe` from Telegram return up-to-date state
without re-hitting eToro.

## Tests

```bash
. .venv/bin/activate
python -m unittest discover -s tests -v
```

Currently 256 unit tests (no network or eToro/Azure access required —
all external calls are stubbed). Per the project rule the test runner
uses the in-tree `.venv`; if you don't have one, `pip install -r
requirements.txt` into a fresh venv first.
