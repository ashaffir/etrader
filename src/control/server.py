"""Tiny stdlib HTTP server exposing the bot's control surface.

We deliberately avoid Flask / FastAPI: zero new dependencies, ~200 LOC,
and the API has six endpoints serving JSON. The server runs in a
daemon thread inside the trading bot process.

Authentication: every request must carry
``Authorization: Bearer <INTERNAL_API_TOKEN>``. Requests without it (or
with a wrong token) get 401. Without ``INTERNAL_API_TOKEN`` set the
server refuses to start at all.

CORS / browsers: not supported by design. This API is for the Telegram
service container only.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .controller import BotController, ControllerError
from .handlers import RouteTable, build_route_table


class _ControlHTTPRequestHandler(BaseHTTPRequestHandler):
    """Per-request handler. Reads body, dispatches via :class:`RouteTable`."""

    server_version = "etrader-control/1.0"

    # injected by the server subclass
    routes: RouteTable
    bearer_token: str
    controller: BotController
    log_adapter: logging.Logger | logging.LoggerAdapter

    # Silence the default `BaseHTTPRequestHandler.log_message` which
    # writes to stderr; route through our logger instead.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        self.log_adapter.debug("[control] %s - %s", self.address_string(), format % args)

    # -- entry points ---------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    # -- core dispatch --------------------------------------------------

    def _handle(self, method: str) -> None:
        if not self._auth_ok():
            self._reply(401, {"error": "unauthorized"})
            return

        path, _, query = self.path.partition("?")
        try:
            handler = self.routes.lookup(method, path)
        except KeyError:
            self._reply(404, {"error": f"unknown route {method} {path}"})
            return

        body: dict[str, Any] | None = None
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            if raw:
                try:
                    body = json.loads(raw.decode("utf-8"))
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                except (UnicodeDecodeError, ValueError) as exc:
                    self._reply(400, {"error": f"invalid JSON body: {exc}"})
                    return
            else:
                body = {}

        try:
            status, payload = handler(self.controller, body, _parse_query(query))
        except ControllerError as exc:
            self._reply(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 — last-resort guard
            self.log_adapter.exception("[control] handler crashed: %s", exc)
            self._reply(500, {"error": f"internal: {exc}"})
            return
        self._reply(status, payload)

    # -- helpers --------------------------------------------------------

    def _auth_ok(self) -> bool:
        header = self.headers.get("Authorization") or ""
        if not header.lower().startswith("bearer "):
            return False
        token = header[len("Bearer "):].strip()
        return bool(token) and token == self.bearer_token

    def _reply(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _parse_query(query: str) -> dict[str, str]:
    if not query:
        return {}
    out: dict[str, str] = {}
    for kv in query.split("&"):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        out[k] = v
    return out


class ControlHTTPServer:
    """Thread wrapper around :class:`ThreadingHTTPServer`."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        bearer_token: str,
        controller: BotController,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        if not bearer_token:
            raise ValueError(
                "Refusing to start the control server without INTERNAL_API_TOKEN. "
                "Set it in .env first."
            )
        self._host = host
        self._port = int(port)
        self._token = bearer_token
        self._controller = controller
        self._log = logger or logging.getLogger("etrader.control.server")

        routes = build_route_table()
        # Build a custom subclass per-instance so the per-request handler
        # can access controller/routes/token without globals.
        class _Handler(_ControlHTTPRequestHandler):
            pass
        _Handler.routes = routes
        _Handler.bearer_token = self._token
        _Handler.controller = self._controller
        _Handler.log_adapter = self._log
        self._handler_cls = _Handler
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self._host, self._port), self._handler_cls)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="etrader-control-http",
            daemon=True,
        )
        self._thread.start()
        self._log.info("[control] listening on http://%s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
        self._httpd = None
        self._thread = None
