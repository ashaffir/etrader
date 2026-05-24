"""etrader — autonomous eToro trading bot.

Package layout:
- :mod:`src.config`         — env + TOML loading + validation
- :mod:`src.logging_setup`  — colored stdout + rotating file logger
- :mod:`src.state`          — in-memory bot state (cooldowns, baselines)
- :mod:`src.etoro`          — eToro Public API client + wrappers
- :mod:`src.ai`             — Azure Foundry / OpenAI chat client + prompts
- :mod:`src.strategy`       — indicators, signals, decisions, universe, risk
- :mod:`src.execution`      — order executor (paper/live) + monitor
- :mod:`src.main`           — entry point
"""

__all__ = [
    "config",
    "logging_setup",
    "state",
]
