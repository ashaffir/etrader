"""Minimal Telegram Bot API client built on ``requests``.

Why not python-telegram-bot / aiogram? Adding a Telegram-specific async
framework would force the rest of the service into asyncio for a tiny
list of API calls. Long-polling is ~30 lines and the only methods we
actually need are ``getUpdates``, ``sendMessage``, plus the callback-
query plumbing we use for the /alerts inline keyboard.

Reference: https://core.telegram.org/bots/api
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests


_BASE = "https://api.telegram.org"


class TelegramApiError(Exception):
    """Raised on transport / HTTP failures."""


# ---------------------------------------------------------------------------
# Update payloads (text messages + inline-keyboard callback queries)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncomingMessage:
    update_id: int
    chat_id: int
    user_id: int
    username: str | None
    text: str
    is_command: bool


@dataclass(frozen=True)
class IncomingCallback:
    """A button-tap on an inline keyboard.

    We only need ``id`` (to ack the tap), ``data`` (the action payload
    we baked into the button), the originating chat/message (so we can
    edit the keyboard in place) and the user (to enforce the same
    allow-list as for text messages).
    """

    update_id: int
    callback_id: str
    chat_id: int
    message_id: int
    user_id: int
    username: str | None
    data: str


# A single getUpdates batch can contain a mix of message and callback
# updates. We surface them in arrival order so the bot loop can replay
# them deterministically.
TelegramUpdate = IncomingMessage | IncomingCallback


@dataclass
class TelegramClient:
    bot_token: str
    timeout_seconds: float = 35.0
    poll_timeout_seconds: int = 30
    session: requests.Session | None = None
    logger: logging.Logger | None = field(default=None)
    _last_update_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if self.session is None:
            self.session = requests.Session()
        if self.logger is None:
            self.logger = logging.getLogger("etrader.telegram.api")

    # -- high-level: incoming -------------------------------------------

    def get_updates(self) -> list[TelegramUpdate]:
        """Long-poll Telegram for new updates; returns mixed messages + callbacks."""
        params: dict[str, Any] = {
            "timeout": int(self.poll_timeout_seconds),
            "allowed_updates": '["message","edited_message","callback_query"]',
        }
        if self._last_update_id:
            params["offset"] = self._last_update_id + 1
        result = self._request("GET", "getUpdates", params=params)
        out: list[TelegramUpdate] = []
        for u in result or []:
            update_id = int(u.get("update_id") or 0)
            if update_id > self._last_update_id:
                self._last_update_id = update_id
            cb = u.get("callback_query")
            if isinstance(cb, dict):
                parsed_cb = _parse_callback(update_id, cb)
                if parsed_cb is not None:
                    out.append(parsed_cb)
                continue
            msg = u.get("message") or u.get("edited_message")
            if isinstance(msg, dict):
                parsed_msg = _parse_message(update_id, msg)
                if parsed_msg is not None:
                    out.append(parsed_msg)
        return out

    # -- high-level: outgoing -------------------------------------------

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = True,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Send a chat message; returns the API result for the LAST chunk.

        The API result includes ``message_id`` which the caller can pin
        in order to later edit the keyboard via :meth:`edit_reply_markup`.
        ``reply_markup`` is only attached to the final chunk so we don't
        confuse the user with multiple identical keyboards.
        """
        if not text:
            return None
        last_result: dict[str, Any] | None = None
        chunks = _split_for_telegram(text, limit=3500)
        for idx, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": int(chat_id),
                "text": chunk,
                "disable_web_page_preview": bool(disable_web_page_preview),
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if reply_markup is not None and idx == len(chunks) - 1:
                payload["reply_markup"] = reply_markup
            last_result = self._request("POST", "sendMessage", body=payload)
        return last_result

    def edit_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        reply_markup: dict[str, Any] | None,
    ) -> None:
        """Replace the inline keyboard on an existing message in place."""
        body: dict[str, Any] = {
            "chat_id": int(chat_id),
            "message_id": int(message_id),
        }
        if reply_markup is not None:
            body["reply_markup"] = reply_markup
        try:
            self._request("POST", "editMessageReplyMarkup", body=body)
        except TelegramApiError as exc:
            # "message is not modified" is a benign no-op on Telegram's
            # side; don't propagate it as a hard error.
            if "not modified" in str(exc).lower():
                return
            raise

    def answer_callback(
        self,
        callback_id: str,
        *,
        text: str | None = None,
        show_alert: bool = False,
    ) -> None:
        """Ack a button tap so the spinner clears on the user's client."""
        body: dict[str, Any] = {"callback_query_id": callback_id}
        if text:
            body["text"] = text[:200]
            body["show_alert"] = bool(show_alert)
        try:
            self._request("POST", "answerCallbackQuery", body=body)
        except TelegramApiError as exc:
            # Old/expired callbacks are common; don't crash the loop over them.
            assert self.logger is not None
            self.logger.debug("answerCallbackQuery failed (ignored): %s", exc)

    # -- low-level ------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        assert self.session is not None
        url = f"{_BASE}/bot{self.bot_token}/{endpoint}"
        try:
            resp = self.session.request(
                method,
                url,
                params=params,
                json=body,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TelegramApiError(f"telegram unreachable: {exc}") from exc
        try:
            payload = resp.json()
        except ValueError as exc:
            raise TelegramApiError(f"telegram non-JSON: {resp.text[:200]}") from exc
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise TelegramApiError(
                f"telegram error: {payload.get('description', 'unknown')}"
            )
        return payload.get("result")


# ---------------------------------------------------------------------------
# Update parsing helpers (kept module-level so tests can hit them directly)
# ---------------------------------------------------------------------------

def _parse_message(update_id: int, msg: dict[str, Any]) -> IncomingMessage | None:
    text = msg.get("text")
    if not isinstance(text, str):
        return None
    chat = msg.get("chat") or {}
    sender = msg.get("from") or {}
    return IncomingMessage(
        update_id=update_id,
        chat_id=int(chat.get("id") or 0),
        user_id=int(sender.get("id") or 0),
        username=sender.get("username"),
        text=text,
        is_command=text.startswith("/"),
    )


def _parse_callback(update_id: int, cb: dict[str, Any]) -> IncomingCallback | None:
    callback_id = str(cb.get("id") or "")
    data = cb.get("data")
    if not callback_id or not isinstance(data, str):
        return None
    msg = cb.get("message") or {}
    chat = msg.get("chat") or {}
    sender = cb.get("from") or {}
    return IncomingCallback(
        update_id=update_id,
        callback_id=callback_id,
        chat_id=int(chat.get("id") or 0),
        message_id=int(msg.get("message_id") or 0),
        user_id=int(sender.get("id") or 0),
        username=sender.get("username"),
        data=data,
    )


def _split_for_telegram(text: str, *, limit: int = 3500) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        parts.append(remaining)
    return parts
