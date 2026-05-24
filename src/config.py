"""Configuration loader.

Reads ``.env`` (key=value lines) and ``config.toml`` (behavior defaults),
validates them against typed dataclasses, and exposes a single
:class:`AppConfig` to the rest of the bot.

The .env parsing is intentionally tiny (no third-party dep): we only need
plain ``KEY=VALUE`` lines, optional surrounding quotes, and comments. The
bot's runtime credentials never leave this module.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import asdict, dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Iterable

from .config_store import PERSISTED_SECTIONS, ConfigStore, open_store


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DB_PATH = "data/config.sqlite"


# ---- .env loading (no third-party dep) ----------------------------------

def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a `.env` file into a dict; skip blanks/comments/malformed."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate `export KEY=...`
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key:
            out[key] = value
    return out


def merge_into_environ(env: dict[str, str], overwrite: bool = False) -> None:
    """Merge env dict into ``os.environ`` (without overwriting by default)."""
    for k, v in env.items():
        if overwrite or k not in os.environ:
            os.environ[k] = v


# ---- Typed config schema ------------------------------------------------

@dataclass
class GuardrailsConfig:
    """Trade caps and safety limits.

    Mutable on purpose: the Telegram control surface lets the operator
    edit individual fields at runtime; replacing fields here propagates
    immediately to RiskEvaluator, TradeExecutor and DecisionEngine
    because they all hold the same instance by reference.
    """

    max_per_trade_usd: float = 500.0
    max_parallel_trades: int = 10
    daily_loss_stop_usd: float = 250.0
    per_instrument_cooldown_min: int = 60
    default_stop_loss_pct: float = 5.0
    default_take_profit_pct: float = 8.0
    max_leverage: int = 1


@dataclass(frozen=True)
class OperationsConfig:
    check_interval_seconds: int = 60
    universe_refresh_minutes: int = 30
    candle_interval: str = "OneHour"
    candle_count: int = 100
    request_timeout_seconds: int = 20
    trade_spacing_seconds: int = 3


@dataclass(frozen=True)
class UniverseConfig:
    base_symbols: tuple[str, ...] = ()
    max_tracked: int = 25
    enable_llm_rotation: bool = True


@dataclass(frozen=True)
class StrategyConfig:
    """Tunables for the price-tool entry / exit ensemble.

    The bot promotes a tracked instrument to a Candidate when a weighted
    sum of price-tool component scores crosses the appropriate threshold.
    Each component emits a signed score in [-1, +1]: +1 for a strong
    bullish trigger, -1 for a strong bearish trigger, smaller magnitudes
    for weaker / state-only signals.

    Combined raw_score is computed as ``sum(weight_i * score_i) /
    sum(|weight_i|)``. ``buy_strength = max(0, raw_score)``,
    ``sell_strength = max(0, -raw_score)``.
    """

    # Indicator periods
    sma_short_period: int = 10
    sma_long_period: int = 30
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0
    donchian_period: int = 20
    momentum_lookback: int = 10

    # Thresholds — calibrated so a typical "3 components vote bullish, the
    # rest neutral" setup clears the entry bar; exits are deliberately
    # easier to trip so the bot cuts losers fast.
    min_signal_strength: float = 0.40   # min positive ensemble score for BUY
    min_exit_strength: float = 0.25     # min negative ensemble score for CLOSE on owned

    # Component weights (set to 0.0 to disable a single component)
    weight_sma_cross: float = 1.0
    weight_ema_cross: float = 0.8
    weight_rsi: float = 0.6
    weight_macd: float = 1.0
    weight_bollinger: float = 0.7
    weight_donchian: float = 0.9
    weight_momentum: float = 0.7


@dataclass(frozen=True)
class AiConfig:
    enabled: bool = True
    max_completion_tokens: int = 4000
    veto_on_unavailable: bool = True
    decision_lookback_candles: int = 30


@dataclass(frozen=True)
class ToolsConfig:
    """Knobs for the tool catalog + selector.

    Defaults match the recommended balanced profile: every built-in
    tool is enabled, the selector keeps up to 14 tools per cycle, and
    a tool needs at least 30 outcomes before its hit-rate is allowed
    to demote it.
    """

    enabled: bool = True
    max_tools_per_cycle: int = 14
    min_hit_rate: float = 0.40
    min_observations: int = 30
    feed_enabled: bool = True
    feed_take: int = 20
    feed_cache_ttl_seconds: int = 600
    regime_anchors: tuple[str, ...] = ("SPX500", "BTC")
    higher_tf_interval: str = "OneDay"
    higher_tf_count: int = 120
    performance_log_path: str = "data/tool_performance.jsonl"
    spread_max_pct: float = 0.5


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/trader.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    color_stdout: bool = True


@dataclass(frozen=True)
class EtoroCredentials:
    public_key: str
    user_key: str
    is_real: bool   # True → use real endpoints; False → demo
    allow_real: bool


@dataclass(frozen=True)
class AzureCredentials:
    endpoint: str | None
    api_key: str | None
    deployment: str | None
    api_version: str = "2024-12-01-preview"
    is_reasoning_model: bool = False

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment)


@dataclass(frozen=True)
class ControlServiceConfig:
    """Local HTTP control API the Telegram service talks to.

    Bound to localhost by default — the Telegram container is expected
    to share the host loopback (or be exposed inside a private network).
    """

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8770
    internal_api_token: str | None = None

    @property
    def is_secured(self) -> bool:
        return bool(self.internal_api_token)


@dataclass(frozen=True)
class AlertingConfig:
    """Trading-bot side of the alerts pipeline.

    The Telegram chat-ID allow-list is shared with the Telegram service
    (same env var) so the bot knows where to fan alerts out without an
    explicit registration handshake. Alerts are no-ops when the allow-
    list is empty, so the trading bot is happy to run without Telegram.
    """

    allowed_chat_ids: tuple[int, ...] = ()
    max_queue_per_chat: int = 200
    subscriptions_file: str = "data/alert_subscriptions.json"

    @property
    def is_enabled(self) -> bool:
        return bool(self.allowed_chat_ids)


@dataclass(frozen=True)
class TelegramConfig:
    """Configuration for the standalone Telegram service.

    Loaded the same way as the trading bot's config so a single .env
    drives both processes. The trading bot never reads these (only the
    `python -m src.telegram_service` entry point does).
    """

    bot_token: str | None
    allowed_chat_ids: tuple[int, ...]
    log_level: str = "INFO"
    poll_timeout_seconds: int = 30
    request_timeout_seconds: int = 35

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token) and bool(self.allowed_chat_ids)


@dataclass(frozen=True)
class AppConfig:
    trading_mode: str  # "paper" | "live"
    guardrails: GuardrailsConfig
    operations: OperationsConfig
    universe: UniverseConfig
    strategy: StrategyConfig
    ai: AiConfig
    tools: ToolsConfig
    logging: LoggingConfig
    etoro: EtoroCredentials
    azure: AzureCredentials
    control: ControlServiceConfig
    alerting: AlertingConfig

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def env_segment(self) -> str:
        """`demo` for paper mode, `real` for live."""
        return "demo" if self.is_paper else "real"


# ---- Loaders ------------------------------------------------------------

# Field names that should always be coerced from list → tuple before being
# fed into a frozen dataclass (the dataclass declares them as tuples, but
# TOML / JSON / SQLite all decode them as lists).
_TUPLE_FIELDS = {"base_symbols", "regime_anchors"}


def _coerce_field_values(kwargs: dict[str, Any]) -> dict[str, Any]:
    for tf in _TUPLE_FIELDS:
        v = kwargs.get(tf)
        if isinstance(v, list):
            kwargs[tf] = tuple(v)
    return kwargs


def _from_dict(cls: Any, data: dict[str, Any] | None) -> Any:
    """Build a dataclass instance from a dict, ignoring unknown keys."""
    data = data or {}
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in data.items() if k in fields}
    return cls(**_coerce_field_values(kwargs))


def _build_section(
    cls: Any,
    *,
    toml: dict[str, Any] | None,
    db: dict[str, Any] | None,
) -> Any:
    """Construct a dataclass section from TOML defaults + DB overrides.

    Layering: dataclass field defaults → TOML → DB. The DB always wins
    when it has a value for a given field. Unknown keys are silently
    dropped so a schema mismatch (e.g. an old DB row for a removed
    field) never crashes startup.
    """
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for source in (toml or {}, db or {}):
        for k, v in source.items():
            if k in fields:
                kwargs[k] = v
    return cls(**_coerce_field_values(kwargs))


def _section_to_dict(instance: Any) -> dict[str, Any]:
    """Return ``{field: jsonable_value}`` for a dataclass instance."""
    out: dict[str, Any] = {}
    for f in dc_fields(instance):
        v = getattr(instance, f.name)
        if isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _truthy(s: str | None) -> bool:
    if not s:
        return False
    return s.strip().lower() in {"1", "true", "yes", "on"}


def load_config(
    project_root: Path = PROJECT_ROOT,
    env_path: Path | None = None,
    config_path: Path | None = None,
    *,
    config_store: ConfigStore | None = None,
    config_db_path: Path | None = None,
    snapshot_on_first_run: bool = True,
) -> AppConfig:
    """Load + validate config from ``.env``, ``config.toml`` and the DB.

    Override layering (highest precedence last):

    1. Dataclass field defaults.
    2. ``config.toml`` (bootstrap defaults; only consulted for fields
       the DB has no row for).
    3. Persisted overrides in ``data/config.sqlite`` — these win.

    On first ever run the DB is empty; the effective merged config is
    snapshotted back so subsequent restarts are DB-authoritative even
    if someone edits ``config.toml`` later.
    """
    env_path = env_path or (project_root / ".env")
    config_path = config_path or (project_root / "config.toml")

    env = load_env_file(env_path)
    merge_into_environ(env, overwrite=False)

    raw = _read_toml(config_path)
    trading_mode = (raw.get("mode") or {}).get("trading", "paper").lower()
    if trading_mode not in {"paper", "live"}:
        raise ValueError(f"config.toml [mode] trading must be 'paper' or 'live', got {trading_mode!r}")

    store, store_owned = _resolve_store(config_store, config_db_path, project_root)
    try:
        guardrails = _build_section(
            GuardrailsConfig, toml=raw.get("guardrails"), db=store.get_section("guardrails"),
        )
        operations = _build_section(
            OperationsConfig, toml=raw.get("operations"), db=store.get_section("operations"),
        )
        universe = _build_section(
            UniverseConfig, toml=raw.get("universe"), db=store.get_section("universe"),
        )
        strategy = _build_section(
            StrategyConfig, toml=raw.get("strategy"), db=store.get_section("strategy"),
        )
        ai = _build_section(
            AiConfig, toml=raw.get("ai"), db=store.get_section("ai"),
        )
        tools = _build_section(
            ToolsConfig, toml=raw.get("tools"), db=store.get_section("tools"),
        )
        logging_cfg = _build_section(
            LoggingConfig, toml=raw.get("logging"), db=store.get_section("logging"),
        )

        if snapshot_on_first_run:
            store.snapshot_if_empty({
                "guardrails": _section_to_dict(guardrails),
                "operations": _section_to_dict(operations),
                "universe":   _section_to_dict(universe),
                "strategy":   _section_to_dict(strategy),
                "ai":         _section_to_dict(ai),
                "tools":      _section_to_dict(tools),
                "logging":    _section_to_dict(logging_cfg),
            })
    finally:
        if store_owned:
            store.close()

    public_key = os.environ.get("PUBLIC_KEY") or os.environ.get("ETORO_API_KEY")
    demo_key = os.environ.get("PRIVATE_KEY") or os.environ.get("DEMO_USER_KEY")
    real_key = os.environ.get("REAL_USER_KEY") or ""
    allow_real = _truthy(os.environ.get("ALLOW_REAL"))

    if not public_key:
        raise RuntimeError("PUBLIC_KEY is required in .env")

    if trading_mode == "live":
        if not allow_real:
            raise RuntimeError("Refusing to run live: set ALLOW_REAL=true in .env first.")
        if not real_key:
            raise RuntimeError("Refusing to run live: REAL_USER_KEY is empty in .env.")
        user_key = real_key
        is_real = True
    else:
        if not demo_key:
            raise RuntimeError("PRIVATE_KEY (demo user-key) is required for paper mode.")
        user_key = demo_key
        is_real = False

    etoro = EtoroCredentials(
        public_key=public_key,
        user_key=user_key,
        is_real=is_real,
        allow_real=allow_real,
    )

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    explicit_reasoning = os.environ.get("AZURE_OPENAI_IS_REASONING_MODEL")
    if explicit_reasoning is None or explicit_reasoning == "":
        is_reasoning = _looks_like_reasoning(deployment)
    else:
        is_reasoning = _truthy(explicit_reasoning)

    azure = AzureCredentials(
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        deployment=deployment,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
        is_reasoning_model=is_reasoning,
    )

    control_cfg_raw = raw.get("control") or {}
    control = ControlServiceConfig(
        enabled=bool(control_cfg_raw.get("enabled", True)),
        host=str(control_cfg_raw.get("host", "127.0.0.1")),
        port=int(control_cfg_raw.get("port", 8770)),
        internal_api_token=os.environ.get("INTERNAL_API_TOKEN") or None,
    )

    alerting_cfg_raw = raw.get("alerting") or {}
    alerting = AlertingConfig(
        allowed_chat_ids=_parse_chat_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")),
        max_queue_per_chat=int(alerting_cfg_raw.get("max_queue_per_chat", 200)),
        subscriptions_file=str(
            alerting_cfg_raw.get("subscriptions_file", "data/alert_subscriptions.json")
        ),
    )

    return AppConfig(
        trading_mode=trading_mode,
        guardrails=guardrails,
        operations=operations,
        universe=universe,
        strategy=strategy,
        ai=ai,
        tools=tools,
        logging=logging_cfg,
        etoro=etoro,
        azure=azure,
        control=control,
        alerting=alerting,
    )


def _parse_chat_ids(raw: str) -> tuple[int, ...]:
    """Comma-separated chat IDs from .env → tuple[int]; lenient on junk."""
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return tuple(out)


@dataclass(frozen=True)
class TelegramServiceRuntimeConfig:
    """Runtime knobs for the Telegram process: where the control API lives."""

    telegram: TelegramConfig
    control_base_url: str
    internal_api_token: str | None


def load_telegram_runtime_config(
    project_root: Path = PROJECT_ROOT,
    env_path: Path | None = None,
) -> TelegramServiceRuntimeConfig:
    """Single config object the Telegram process needs to start up."""
    tg = load_telegram_config(project_root=project_root, env_path=env_path)
    base = os.environ.get("CONTROL_API_BASE_URL")
    if not base:
        host = os.environ.get("CONTROL_API_HOST") or "127.0.0.1"
        port = _safe_int(os.environ.get("CONTROL_API_PORT"), default=8770)
        base = f"http://{host}:{port}"
    return TelegramServiceRuntimeConfig(
        telegram=tg,
        control_base_url=base.rstrip("/"),
        internal_api_token=os.environ.get("INTERNAL_API_TOKEN") or None,
    )


def load_telegram_config(
    project_root: Path = PROJECT_ROOT,
    env_path: Path | None = None,
) -> TelegramConfig:
    """Load Telegram-only credentials. Used by ``python -m src.telegram_service``.

    Kept separate from :func:`load_config` so the Telegram service does
    NOT need to validate the bot's eToro / Azure credentials, and won't
    refuse to start just because (e.g.) ``REAL_USER_KEY`` is missing.
    """
    env_path = env_path or (project_root / ".env")
    env = load_env_file(env_path)
    merge_into_environ(env, overwrite=False)

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or None
    raw_ids = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
    chat_ids: list[int] = []
    for part in raw_ids.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            chat_ids.append(int(part))
        except ValueError:
            continue
    log_level = (os.environ.get("TELEGRAM_LOG_LEVEL") or "INFO").upper()
    poll_timeout = _safe_int(os.environ.get("TELEGRAM_POLL_TIMEOUT_SECONDS"), default=30)
    return TelegramConfig(
        bot_token=token,
        allowed_chat_ids=tuple(chat_ids),
        log_level=log_level,
        poll_timeout_seconds=poll_timeout,
        request_timeout_seconds=poll_timeout + 5,
    )


def _resolve_store(
    explicit: ConfigStore | None,
    db_path: Path | None,
    project_root: Path,
) -> tuple[ConfigStore, bool]:
    """Return ``(store, owned)``.

    If a caller supplied a store, use it as-is and don't take ownership
    (they'll close it). Otherwise open a fresh store rooted at the
    project; the caller-facing :func:`load_config` will close it once
    the AppConfig has been materialised.
    """
    if explicit is not None:
        return explicit, False
    path = db_path or (project_root / DEFAULT_CONFIG_DB_PATH)
    return open_store(path), True


def _safe_int(s: str | None, *, default: int) -> int:
    if not s:
        return default
    try:
        return int(s.strip())
    except ValueError:
        return default


def _looks_like_reasoning(name: str | None) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    return any(n.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))


def summarize_config(cfg: AppConfig) -> str:
    """Human-readable one-paragraph summary, safe to log (no secrets)."""
    parts: Iterable[str] = (
        f"mode={cfg.trading_mode}",
        f"env_segment=/{cfg.env_segment}/",
        f"per_trade_cap=${cfg.guardrails.max_per_trade_usd:.0f}",
        f"max_parallel={cfg.guardrails.max_parallel_trades}",
        f"daily_loss_stop=${cfg.guardrails.daily_loss_stop_usd:.0f}",
        f"check_every={cfg.operations.check_interval_seconds}s",
        f"universe_base={len(cfg.universe.base_symbols)}",
        f"ai_enabled={cfg.ai.enabled and cfg.azure.is_configured}",
    )
    return ", ".join(parts)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    try:
        print(summarize_config(load_config()))
    except Exception as exc:  # noqa: BLE001 - diag-only
        print(f"config error: {exc}", file=sys.stderr)
        sys.exit(1)
