"""Standalone Telegram service.

A separate process from the trading bot. It polls the Telegram Bot API
for new messages, dispatches commands, and forwards them to the
trading bot's internal HTTP control API (auth'd by INTERNAL_API_TOKEN).

Run with:

    python -m src.telegram_service

The process exits cleanly on Ctrl+C / SIGTERM.
"""
