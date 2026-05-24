# etrader — autonomous eToro trading bot

A Python trading bot that uses the **eToro Public API** for market data and
order execution and an **Azure AI Foundry / Azure OpenAI** deployment as a
decision overlay on top of deterministic technical signals.

The bot:

1. Loads tracked instruments (a curated baseline + optional LLM-suggested
   rotations) into a refreshable universe.
2. Pulls live prices and OHLCV candles every cycle.
3. Computes deterministic indicators (SMA cross, RSI, momentum) into a
   shortlist of BUY / CLOSE candidates.
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

Everything is logged to stdout (colored) and to a rotating file.

## Layout

```
etrader/
├── .env                       # eToro + Azure + Telegram credentials
├── config.toml                # behaviour defaults — guardrails, ops, universe, AI, control
├── requirements.txt           # runtime deps: requests, openai
├── src/
│   ├── main.py                # trading bot entry: `python -m src.main`
│   ├── config.py              # .env + TOML loader, schema validation
│   ├── logging_setup.py       # colored stdout + rotating file logger
│   ├── state.py               # in-memory bot state (cooldowns, baseline, owned IDs)
│   ├── persistence.py         # save/load BotState to data/bot_state.json
│   ├── trade_history.py       # append-only data/trade_history.jsonl
│   ├── telemetry.py           # in-memory snapshot store (read by Telegram)
│   ├── etoro/                 # API client + endpoint wrappers
│   ├── ai/                    # Azure Foundry chat client + prompts (incl. Q&A)
│   ├── strategy/              # indicators, signals, decisions, universe, risk
│   ├── execution/             # executor (paper/live) + position monitor
│   ├── control/               # internal HTTP control API (consumed by Telegram)
│   │   ├── controller.py      # thread-safe pause/resume/panic/ask facade
│   │   ├── server.py          # stdlib HTTP server with bearer-token auth
│   │   └── handlers.py        # JSON endpoint dispatch table
│   └── telegram_service/      # SEPARATE PROCESS: Telegram bot poller + dispatcher
│       ├── __main__.py        # entry: `python -m src.telegram_service`
│       ├── bot.py             # long-polling loop
│       ├── commands.py        # parse + dispatch /commands and free-text
│       ├── control_client.py  # requests-based HTTP client for src.control
│       ├── telegram_api.py    # raw Bot API calls (getUpdates, sendMessage)
│       └── formatters.py      # render JSON responses as Telegram text
└── tests/                     # unit tests (stdlib unittest)
```

## Setup

```bash
# 1. Make sure .env is populated (PUBLIC_KEY, PRIVATE_KEY, AZURE_*).
# 2. Create a venv and install deps.
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 3. Run the bot
python -m src.main

# 4. Run tests
python -m unittest discover -s tests -v
```

## Mode

`config.toml`'s `[mode] trading` toggles paper vs. live:

- `paper` (default) → demo environment, fake balance, real prices. Safe.
- `live` → real environment. Requires `ALLOW_REAL=true` AND `REAL_USER_KEY`
  in `.env`. The bot refuses to start otherwise.

## Guardrails

| Limit                            | Default | Where to change         |
|----------------------------------|---------|-------------------------|
| Max cash per trade               | $500    | `config.toml` `[guardrails]` |
| Max parallel positions (bot-owned) | 10    | same                     |
| Daily-loss kill switch           | $250    | same                     |
| Per-instrument cooldown          | 60 min  | same                     |
| Default stop-loss                | 5%      | same                     |
| Default take-profit              | 8%      | same                     |

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
[universe ] tracking 12 instruments (base=10, llm=2)
[market   ] fetched 12 rates, 12 candle sets in 410 ms
[signals  ] 3 buy / 0 close candidates (NVDA, MSFT, BTC)
[ai       ] decision: BUY NVDA 250 USD; HOLD others (latency 1.4 s)
[risk     ] approved 1 / 1 — within all caps
[exec     ] OPEN  NVDA  250.00 USD  long  SL=128.40 TP=143.10  → orderID 13902598
[portfolio] equity=$10,251.83  available=$9,500.83  open=4
─────────────────────────────────────
```

DEBUG level adds raw JSON envelopes per call.
