"""Entry point for the standalone Telegram service.

Run with::

    python -m src.telegram_service

Reads ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_ALLOWED_CHAT_IDS`` from
``.env``; talks to the trading bot over the control HTTP API
described in :mod:`src.control.handlers`. The trading bot must be
running and reachable (default ``http://127.0.0.1:8770``).
"""

from __future__ import annotations

import logging
import signal
import sys

from ..config import load_telegram_runtime_config
from .bot import TelegramService
from .control_client import ControlAPIClient
from .telegram_api import TelegramClient


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    cfg = load_telegram_runtime_config()
    _configure_logging(cfg.telegram.log_level)
    log = logging.getLogger("etrader.telegram")

    if not cfg.telegram.is_configured:
        log.error(
            "Telegram service cannot start: TELEGRAM_BOT_TOKEN and/or "
            "TELEGRAM_ALLOWED_CHAT_IDS are missing in .env"
        )
        return 2
    if not cfg.internal_api_token:
        log.error(
            "Telegram service cannot start: INTERNAL_API_TOKEN missing in .env "
            "(required to authenticate against the trading bot's control API)"
        )
        return 2

    telegram = TelegramClient(
        bot_token=cfg.telegram.bot_token or "",
        timeout_seconds=cfg.telegram.request_timeout_seconds,
        poll_timeout_seconds=cfg.telegram.poll_timeout_seconds,
        logger=logging.getLogger("etrader.telegram.api"),
    )
    api = ControlAPIClient(
        base_url=cfg.control_base_url,
        token=cfg.internal_api_token,
        logger=logging.getLogger("etrader.telegram.control_client"),
    )
    service = TelegramService(
        telegram=telegram,
        api=api,
        allowed_chat_ids=cfg.telegram.allowed_chat_ids,
        logger=log,
    )

    def _stop(*_: object) -> None:
        log.info("telegram service shutting down")
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        return service.run() or 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
