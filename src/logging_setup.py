"""Logging setup — colored console + rotating file.

We deliberately stay on the Python stdlib (``logging``) so the bot has
zero external logging dependencies. Colors use ANSI escape codes
gated by ``isatty()`` so they never pollute log files.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .config import LoggingConfig


# ---------------------------------------------------------------------------
# Color handling
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[37m",     # bright black / grey
    logging.INFO: "\033[36m",      # cyan
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[1;41m",  # bold red bg
}

_TAG_COLOR = "\033[35m"   # magenta for the [tag] prefix
_TIME_COLOR = "\033[90m"  # dim grey for timestamps


class _PrettyFormatter(logging.Formatter):
    """Compact, human-friendly formatter.

    Output shape:
        2026-05-24 11:36:00 INFO     [tag      ] message
    """

    DEFAULT_TAG_WIDTH = 9

    def __init__(self, *, color: bool, tag_width: int = DEFAULT_TAG_WIDTH) -> None:
        super().__init__()
        self.color = color
        self.tag_width = tag_width

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_name = record.levelname.ljust(8)
        tag = getattr(record, "tag", record.name.split(".")[-1])
        tag_str = f"[{str(tag).ljust(self.tag_width)}]"
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        if not self.color:
            return f"{ts} {level_name} {tag_str} {msg}"
        level_color = _LEVEL_COLORS.get(record.levelno, "")
        return (
            f"{_TIME_COLOR}{ts}{_RESET} "
            f"{level_color}{level_name}{_RESET} "
            f"{_TAG_COLOR}{tag_str}{_RESET} "
            f"{msg}"
        )


# ---------------------------------------------------------------------------
# Setup entry-point
# ---------------------------------------------------------------------------

def configure_logging(cfg: LoggingConfig, project_root: Path) -> logging.Logger:
    """Apply ``cfg`` to the root logger; return the bot's root logger."""
    root = logging.getLogger()
    # Reset any prior handlers (idempotent across calls / tests)
    for h in list(root.handlers):
        root.removeHandler(h)

    level = getattr(logging, cfg.level.upper(), logging.INFO)
    root.setLevel(level)

    color_capable = cfg.color_stdout and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(_PrettyFormatter(color=color_capable))
    stdout.setLevel(level)
    root.addHandler(stdout)

    log_path = project_root / cfg.file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rotating = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    rotating.setFormatter(_PrettyFormatter(color=False))
    rotating.setLevel(level)
    root.addHandler(rotating)

    # Tame noisy libraries.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger("etrader")


def get_logger(name: str, tag: str | None = None) -> logging.LoggerAdapter:
    """Return a ``LoggerAdapter`` that adds a `tag` field for the formatter."""
    logger = logging.getLogger(f"etrader.{name}")
    return logging.LoggerAdapter(logger, {"tag": tag or name})
