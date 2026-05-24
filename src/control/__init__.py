"""Internal control surface: thread-safe controller + tiny HTTP server.

The controller is the single object the trading bot, the cycle loop,
and the HTTP handlers all share. It is the only place that mutates
:class:`~src.state.BotState`, persists to disk, or coordinates a panic
close. The HTTP server (server.py + handlers.py) just translates JSON
requests into controller method calls.
"""

from .controller import BotController, ControllerError, ControllerStatus

__all__ = ["BotController", "ControllerError", "ControllerStatus"]
