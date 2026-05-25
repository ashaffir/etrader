# etrader — autonomous eToro trading bot

A Python trading bot that uses the **eToro Public API** for market data and
order execution and an **Azure AI Foundry / Azure OpenAI** deployment as a
decision overlay on top of deterministic technical signals.

The bot:

1. Builds a **news-driven universe** every `[operations] universe_refresh_minutes`.
   A pluggable news pipeline (StockTwits trending, SEC EDGAR 8-Ks, Google News
   RSS, Yahoo RSS, yfinance) folds new observations into `data/news_candidates.json`;
   the universe builder probes each candidate's live ATR% + spread% and admits
   only the tradeable ones. Bot-owned positions are always kept in the universe.
2. Enriches each tracked symbol with cached **fundamentals** (sector, P/E,
   margins, growth, analyst target) so the LLM and `/fundamentals` have
   structural context to weight against pure price action. See
   `data/fundamentals_cache.json`.
3. Pulls live prices and OHLCV candles every cycle.
4. Runs a **weighted ensemble** of price-action indicators (SMA cross,
   EMA cross, RSI, MACD, Bollinger, Donchian, momentum) and produces a
   shortlist of BUY / CLOSE candidates whenever the combined score
   crosses the configurable entry / exit thresholds.
5. Runs an extensible **tool catalog** (~18 tools across price / volume /
   context families) against each candidate. A regime-aware selector picks
   the relevant subset per (instrument, cycle); hard gates like
   `spread_filter` and `market_hours` can veto a candidate before any LLM
   call. See `/signals` from Telegram for the live rule set.
6. Sends the surviving shortlist plus full portfolio state, tool features
   and per-symbol fundamentals to the LLM, which returns a structured JSON
   action plan.
7. Applies guardrails (cap, parallel limit, cooldown, daily-loss stop, paper
   gate) and either executes or simulates the trades.
8. Verifies trades by re-reading the portfolio after the eToro 10s cache.
9. Persists state (`data/bot_state.json`), pushes alerts, and snapshots
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
├── config.toml                # behaviour defaults — guardrails, ops, universe, news, fundamentals, strategy, tools, AI, control, alerting
├── requirements.txt           # runtime deps: requests, openai, yfinance, feedparser
├── run.sh                     # interactive start/stop/restart/status wrapper for the two services
├── data/                      # bot-managed runtime state (gitignored)
│   ├── bot_state.json         # cooldowns, owned positions, daily-loss baseline, paused flag
│   ├── config.sqlite          # persisted overrides for any [guardrails]/etc. edited at runtime
│   ├── instrument_cache.json  # symbol → instrumentID resolution cache
│   ├── news_candidates.json   # news-driven universe candidate store (TTL'd, score-ranked)
│   ├── sec_cik_to_ticker.json # cached SEC EDGAR CIK → ticker mapping (only when sec_edgar source is enabled)
│   ├── fundamentals_cache.json # per-symbol yfinance fundamentals cache (24h refresh + earnings-aware)
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
│   ├── news/                  # news pipeline (sources, aggregator, scheduler, candidate store, channel probe)
│   ├── fundamentals/          # yfinance-backed per-symbol fundamentals cache + LLM projection
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
│       ├── channel_formatter.py # /channels overview / test / logs renderers
│       ├── control_client.py  # requests-based HTTP client for src.control
│       ├── telegram_api.py    # raw Bot API: getUpdates, sendMessage, callback queries
│       └── formatters.py      # render JSON responses as Telegram text
└── tests/                     # unit tests (stdlib unittest, 380+ cases)
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

## Process management — `./run.sh`

`run.sh` is an interactive shell wrapper around the two processes
(`src.main` and `src.telegram_service`). It tracks each service's PID
under a dotfile, streams stdout/stderr into `logs/`, and pings the
control HTTP API for a quick health overview.

Run with no arguments to open the colored menu, or pass a subcommand
to drive everything from CI / scripts:

```bash
./run.sh setup                  # create .venv + install requirements.txt
./run.sh start                  # bot + telegram
./run.sh start bot              # only the trading loop
./run.sh restart telegram       # bounce just the Telegram poller
./run.sh stop                   # both
./run.sh status                 # PIDs, control-API health, last trader log lines
./run.sh logs bot 200 true      # tail logs/bot.out.log (last 200 lines, follow)
./run.sh logs trader 200 true   # tail the rotating logs/trader.log
./run.sh test                   # python -m unittest discover -s tests
./run.sh clean                  # remove launch logs / PIDs (interactive)
```

Service targets are `bot` (trading loop, `python -m src.main`),
`telegram` (`python -m src.telegram_service`) or `all` (default).
Each service writes its nohup output to `logs/bot.out.log` /
`logs/telegram.out.log`; the structured trader log keeps rotating to
`logs/trader.log` as always.

Environment overrides:

| Variable                 | Default       | What it changes                                        |
|--------------------------|---------------|--------------------------------------------------------|
| `ETRADER_CONTROL_HOST`   | `127.0.0.1`   | Host the script probes for the control API in `status` |
| `ETRADER_CONTROL_PORT`   | `8770`        | Port the script probes                                 |
| `INTERNAL_API_TOKEN`     | from `.env`   | Bearer token used to call `/ping` during `status`      |

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
`[universe]`, `[news]`, `[fundamentals]`, `[strategy]`, `[ai]`, `[tools]`,
`[logging]`. Secrets and the `[control]` / `[alerting]` plumbing remain
`.env` / TOML-only.

When a new section is added in a future release (e.g. `[fundamentals]`
in Phase 3), the bot **auto-backfills** TOML defaults for it on the
next boot without touching any sections the operator has already edited.
You never need to delete `data/config.sqlite` to pick up new defaults.

## Universe, news, and fundamentals

The tracked-symbol universe is **news-driven**: nothing is statically
pinned (`[universe] base_symbols = []` by default), and every symbol
admitted to the tracked set is justified by either a news observation
or an open bot-owned position. A typical refresh looks like this:

1. The **news pipeline** (`[news] scan_interval_minutes`, default 60)
   polls every enabled source — **StockTwits trending** (no key),
   **SEC EDGAR 8-Ks** (needs `SEC_USER_AGENT`; see below),
   **Google News RSS** (no key), **Yahoo Finance RSS** (no key) and
   **yfinance.Ticker(...).news** (no key). Free-text headlines are
   ticker-extracted; the universe builder consumes the resulting
   score-ranked `data/news_candidates.json`.
2. The **universe builder** probes the top
   `[universe] probe_max_candidates` (default 50) news candidates for
   live ATR% and spread%, and admits only those that pass the
   **activity filter** (`min_atr_pct`, `max_spread_pct`). Rejected
   candidates are recorded with a reason and surfaced via `/universe`
   and the opt-in `universe_rejected` alert. Bot-owned positions
   always pass (we never want to lose sight of an open trade).
3. The **fundamentals cache** (`[fundamentals]`, default on) tops up
   `data/fundamentals_cache.json` for the tracked symbols on every
   refresh. Each entry is refreshed every `refresh_after_hours`
   (default 24 h) or as soon as its recorded next-earnings timestamp
   passes — quarterly results reset valuations and margins overnight.
   `budget_per_refresh` (default 8) caps how many *stale* symbols
   we'll re-fetch in a single cycle so a 50-ticker universe doesn't
   block on 50 yfinance calls.
4. When the LLM decision call runs, each candidate gets a trim
   fundamentals dict appended (sector, P/E, growth, analyst target,
   …). The LLM is instructed to treat fundamentals as **advisory** —
   they MAY downgrade conviction on a technically-strong candidate
   but never promote a candidate the price ensemble didn't flag. Set
   `[fundamentals] enrich_decision_prompt = false` to keep the cache
   warm without paying the prompt-token cost.

Telegram surfaces:

- `/universe` — currently tracked symbols, the reason each one was
  admitted, and recent activity-filter rejections.
- `/news [N]` — top-N news candidates with their score, sources and
  freshest headline; also shows last-scan stats and the next-run ETA.
- `/channels` — per-source health overview: which feeds are enabled
  in `[news] enabled_sources`, which actually got wired, their
  effective weight, the items kept on the last scan, and any
  self-reported disabled reasons (e.g. SEC EDGAR with no
  `SEC_USER_AGENT`).
- `/channels test [names]` — live one-shot dry-run against each
  source. Results are **never** folded into the candidate store, so
  it's safe to run repeatedly. Pass a comma-separated list to probe
  a subset, e.g. `/channels test stocktwits,yfinance`.
- `/channels logs` — the most recent aggregator run's per-source
  item-kept counts and full per-source error strings.
- `/fundamentals` — list of every cached symbol grouped by sector.
- `/fundamentals <SYM>` — full detail view for one symbol.

### SEC EDGAR — required header

The SEC enforces a User-Agent policy for programmatic access. To
enable the `sec_edgar` news source, set the following in `.env`
(both pieces are required by SEC):

```
SEC_USER_AGENT="YourCompany Trading Bot research@example.com"
```

If `SEC_USER_AGENT` is missing or doesn't contain an `@`, the source
is **disabled at boot** with a warning — the rest of the news
pipeline continues without it. There is no API key, only this
contact-bearing User-Agent.

### Free APIs only

The Phase 2 + 3 news + fundamentals stack is built entirely from
**free, key-less** providers:

| Source                | Auth needed                | Limits          |
|-----------------------|----------------------------|-----------------|
| StockTwits trending   | none                       | ~lenient        |
| SEC EDGAR 8-Ks        | `SEC_USER_AGENT` (no key)  | 10 req/s / IP   |
| Google News RSS       | none                       | rate-limited at IP level if abused |
| Yahoo Finance RSS     | none                       | rate-limited at IP level if abused |
| yfinance (news + info)| none                       | scraping, may rate-limit on bursts |

You don't need to register or generate keys for any of these to run
the bot. If you want commercial-grade SLAs (Bloomberg, Refinitiv,
AlphaVantage, IEX Cloud, etc.) you can plug them in as additional
news sources later — the source interface in `src/news/sources/base.py`
is a single-method Protocol.

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
| `/universe`                   | Instruments tracked, why each, and rejections       |
| `/news [N]`                   | Top-N news candidates + last scan stats             |
| `/channels` (`/sources`)      | Per-source health overview (status, weights, counts) |
| `/channels test [names]`      | Live dry-run against every (or a subset of) source(s) — no DB writes |
| `/channels logs`              | Most-recent scan stats + per-source errors          |
| `/fundamentals [SYM]`         | Cached fundamentals (list or per-symbol detail)     |
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
| `universe_changed`     | News-driven rotation adds/removes instruments       |
| `universe_rejected`    | One or more news candidates failed the activity filter (opt-in) |
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
[news]        scan kept 87 items, +52 obs (stocktwits=40, google_news=37, yfinance=10, …)
[universe]    refreshed → tracking 12 instrument(s) (news=10, owned=2, rejected=5)
[fundamentals] refreshed=8 failed=0 skipped=4 (cache size 17)
[market]      fetched 12/12 rate(s)
[signals]     3 candidate(s): BUY NVDA(0.62), BUY MSFT(0.48), CLOSE TSLA(-0.31)
[regime]      SPX=up(+3.2%), BTC=range
[tools]       1 gated: AAPL(spread_filter)
[ai]          decision (llm, 1432 ms): BUY NVDA 250 USD; HOLD others
[risk]        approved 1 / 1 — all clear
[exec]        OK         BUY    NVDA      — orderID 13902598
[portfolio]   equity=$10,251.83  available=$9,500.83  invested=$751.00  pnl=$+12.40  bot_owned=4
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

Currently 400+ unit tests (no network or eToro/Azure access required —
all external calls are stubbed). Per the project rule the test runner
uses the in-tree `.venv`; if you don't have one, `pip install -r
requirements.txt` into a fresh venv first. `./run.sh test` is a
convenience wrapper that activates the venv first.
