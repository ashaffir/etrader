"""Long-poll loop tying Telegram updates to control-API commands.

This is the heart of the standalone Telegram service: pull updates,
filter by allowed chat ID, dispatch the parsed command, send the
reply. Everything is sequential (one chat at a time) — Telegram's
update guarantees and our small command set don't justify
asyncio/threads here.

In addition to text messages, the loop also handles inline-keyboard
callback queries (currently only the /alerts submenu) and drains the
trading bot's per-chat alert queue each tick so emitted alerts get
forwarded to the operator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .alerts_menu import (
    build_alerts_caption,
    build_alerts_keyboard,
    build_closed_caption,
    is_alerts_callback,
    parse_alerts_callback,
)
from .commands import CommandContext, dispatch, parse_command
from .control_client import ControlAPIClient, ControlAPIError
from .telegram_api import (
    IncomingCallback,
    IncomingMessage,
    TelegramApiError,
    TelegramClient,
    TelegramUpdate,
)


@dataclass
class TelegramService:
    telegram: TelegramClient
    api: ControlAPIClient
    allowed_chat_ids: tuple[int, ...]
    logger: logging.Logger

    def __post_init__(self) -> None:
        if not self.allowed_chat_ids:
            self.logger.warning(
                "TELEGRAM_ALLOWED_CHAT_IDS is empty — every chat will be rejected. "
                "DM the bot once and add your chat_id to the env var."
            )

    # ------------------------------------------------------------------

    def run(self) -> int:
        self.logger.info(
            "telegram service ready — allowed chats: %s",
            ", ".join(str(c) for c in self.allowed_chat_ids) or "<none>",
        )
        # Best-effort startup ping so we fail loudly if the control API
        # isn't reachable (typo'd port, missing INTERNAL_API_TOKEN, etc.)
        try:
            self.api.ping()
            self.logger.info("control API reachable at %s", self.api.base_url)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("control API ping failed (will keep trying): %s", exc)

        backoff = 1
        while True:
            try:
                updates = self.telegram.get_updates()
            except TelegramApiError as exc:
                self.logger.warning("telegram poll failed: %s (retry in %ds)", exc, backoff)
                time.sleep(backoff)
                backoff = min(60, backoff * 2)
                continue
            backoff = 1
            for upd in updates:
                self._dispatch_update(upd)
            self._drain_alerts_for_all_chats()

    # ------------------------------------------------------------------
    # Update routing
    # ------------------------------------------------------------------

    def _dispatch_update(self, upd: TelegramUpdate) -> None:
        if isinstance(upd, IncomingMessage):
            self._handle_message(upd)
        elif isinstance(upd, IncomingCallback):
            self._handle_callback(upd)

    def _handle_message(self, msg: IncomingMessage) -> None:
        if not self._is_allowed(msg.chat_id):
            self.logger.warning(
                "rejected message from chat_id=%d user=@%s text=%r",
                msg.chat_id, msg.username, msg.text[:120],
            )
            try:
                self.telegram.send_message(
                    msg.chat_id,
                    "This bot is private. Your chat_id is not in the allow-list.",
                )
            except TelegramApiError as exc:
                self.logger.warning("could not reply to rejected chat: %s", exc)
            return

        cmd = parse_command(msg.text)
        ctx = CommandContext(
            api=self.api,
            cmd=cmd,
            sender_username=msg.username,
            logger=self.logger,
            chat_id=msg.chat_id,
        )
        self.logger.info(
            "[chat=%d user=@%s] /%s args=%r", msg.chat_id, msg.username, cmd.name, cmd.args[:120],
        )
        reply = dispatch(ctx)
        try:
            self.telegram.send_message(
                msg.chat_id,
                reply.text,
                reply_markup=reply.reply_markup,
            )
        except TelegramApiError as exc:
            self.logger.warning("send_message failed: %s", exc)

    # ------------------------------------------------------------------
    # Inline keyboard callbacks (currently /alerts only)
    # ------------------------------------------------------------------

    def _handle_callback(self, cb: IncomingCallback) -> None:
        if not self._is_allowed(cb.chat_id):
            self.logger.warning(
                "rejected callback from chat_id=%d user=@%s data=%r",
                cb.chat_id, cb.username, cb.data,
            )
            self.telegram.answer_callback(cb.callback_id, text="Not allowed.")
            return

        if not is_alerts_callback(cb.data):
            self.logger.info(
                "ignoring unknown callback data=%r from chat=%d",
                cb.data, cb.chat_id,
            )
            self.telegram.answer_callback(cb.callback_id)
            return

        parsed = parse_alerts_callback(cb.data)
        if parsed.action == "toggle":
            self._handle_alerts_toggle(cb, type_str=parsed.alert_type)
        elif parsed.action == "close":
            self._handle_alerts_close(cb)
        else:
            self.telegram.answer_callback(cb.callback_id, text="Unknown action.")

    def _handle_alerts_toggle(self, cb: IncomingCallback, *, type_str: str) -> None:
        try:
            result = self.api.toggle_alert_subscription(cb.chat_id, type_str)
        except ControlAPIError as exc:
            self.logger.warning("alerts toggle failed: %s", exc)
            self.telegram.answer_callback(cb.callback_id, text=f"Error: {exc}")
            return

        try:
            payload = self.api.alert_subscriptions(cb.chat_id)
        except ControlAPIError as exc:
            self.logger.warning("alerts subscriptions fetch failed: %s", exc)
            self.telegram.answer_callback(cb.callback_id, text=f"Error: {exc}")
            return

        try:
            self.telegram.edit_reply_markup(
                cb.chat_id,
                cb.message_id,
                build_alerts_keyboard(payload),
            )
        except TelegramApiError as exc:
            self.logger.warning("edit_reply_markup failed: %s", exc)

        new_state = "ON" if result.get("enabled") else "OFF"
        self.telegram.answer_callback(
            cb.callback_id,
            text=f"{result.get('type')} → {new_state}",
        )

    def _handle_alerts_close(self, cb: IncomingCallback) -> None:
        # Strip the keyboard so the menu collapses; replace caption with
        # a one-liner of currently enabled alerts so the operator has a
        # record of the choice.
        try:
            payload = self.api.alert_subscriptions(cb.chat_id)
        except ControlAPIError as exc:
            self.logger.warning("alerts subscriptions fetch failed: %s", exc)
            payload = {"available": []}
        try:
            self.telegram.edit_reply_markup(cb.chat_id, cb.message_id, None)
        except TelegramApiError as exc:
            self.logger.warning("edit_reply_markup (close) failed: %s", exc)
        # We can't edit the original caption text via editMessageReplyMarkup,
        # so post a fresh confirmation message that reflects the saved state.
        try:
            self.telegram.send_message(cb.chat_id, build_closed_caption(payload))
        except TelegramApiError as exc:
            self.logger.warning("send confirmation after /alerts close failed: %s", exc)
        self.telegram.answer_callback(cb.callback_id, text="Saved.")

    # ------------------------------------------------------------------
    # Alert delivery (drain trading-bot queue → forward to chats)
    # ------------------------------------------------------------------

    def _drain_alerts_for_all_chats(self) -> None:
        for chat_id in self.allowed_chat_ids:
            try:
                payload = self.api.alert_pending(chat_id, limit=50)
            except ControlAPIError as exc:
                # Trading bot might be down; quiet log + try again next tick.
                self.logger.debug("alert drain failed for chat=%d: %s", chat_id, exc)
                continue
            for alert in payload.get("alerts") or []:
                title = str(alert.get("title") or "")
                body = str(alert.get("body") or "")
                ts = str(alert.get("timestamp") or "")
                text = self._format_alert(title=title, body=body, timestamp=ts)
                try:
                    self.telegram.send_message(chat_id, text)
                except TelegramApiError as exc:
                    self.logger.warning(
                        "alert send failed (chat=%d): %s", chat_id, exc,
                    )

    @staticmethod
    def _format_alert(*, title: str, body: str, timestamp: str) -> str:
        head = f"[ALERT] {title}" if title else "[ALERT]"
        suffix = f"  ({timestamp})" if timestamp else ""
        if body:
            return f"{head}{suffix}\n{body}"
        return f"{head}{suffix}"

    # ------------------------------------------------------------------

    def _is_allowed(self, chat_id: int) -> bool:
        if not self.allowed_chat_ids:
            return False
        return int(chat_id) in self.allowed_chat_ids
