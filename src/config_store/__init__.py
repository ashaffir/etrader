"""Persistent override store for runtime-tunable config.

The trading bot loads its bootstrap defaults from ``config.toml`` once,
then layers on top of them any overrides that have been written to a
small SQLite database (``data/config.sqlite``). Whenever the operator
edits a guardrail through the Telegram service, the new value is
written back to the database, so the next restart sees it instead of
the TOML default.

Public surface intentionally narrow — the rest of the bot only needs
the :class:`ConfigStore` instance.
"""

from .store import (
    PERSISTED_SECTIONS,
    ConfigStore,
    open_store,
)

__all__ = ["ConfigStore", "PERSISTED_SECTIONS", "open_store"]
