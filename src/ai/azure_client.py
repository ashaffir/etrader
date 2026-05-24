"""Azure AI Foundry / Azure OpenAI chat-completions wrapper.

Imports are lazy: the bot can run without ``openai`` installed if AI
features are disabled (``[ai] enabled = false`` in config). The wrapper
also tolerates reasoning-model deployments (gpt-5 / o-series) by
omitting ``temperature`` and using ``max_completion_tokens``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ..config import AzureCredentials


class AzureUnavailable(Exception):
    """The ``openai`` package isn't installed or credentials are missing."""


@dataclass(frozen=True)
class AiCallResult:
    text: str
    parsed_json: Any | None
    latency_ms: int


class AzureFoundryClient:
    """Stateless wrapper over ``openai.AzureOpenAI``.

    Each call is independent; we hold no chat history because trading
    decisions are stateless prompts (current snapshot in, decision out).
    """

    def __init__(
        self,
        credentials: AzureCredentials,
        *,
        max_completion_tokens: int = 4000,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        if not credentials.is_configured:
            raise AzureUnavailable("Azure credentials are not fully configured.")
        try:
            from openai import AzureOpenAI  # noqa: WPS433 - lazy import
        except ImportError as exc:  # pragma: no cover - import-time failure
            raise AzureUnavailable("openai package not installed") from exc

        self._creds = credentials
        self._max_tokens = int(max_completion_tokens)
        self._logger = logger or logging.getLogger("etrader.ai.azure")
        self._client = AzureOpenAI(
            azure_endpoint=credentials.endpoint,
            api_key=credentials.api_key,
            api_version=credentials.api_version,
        )

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        require_json: bool = True,
    ) -> AiCallResult:
        """Send a single chat turn; parse a JSON object from the response.

        Returns
        -------
        AiCallResult
            ``parsed_json`` is ``None`` when ``require_json`` is False or
            when the response wasn't valid JSON. The raw ``text`` is
            always populated for logging.
        """
        import time

        kwargs: dict[str, Any] = {
            "model": self._creds.deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._creds.is_reasoning_model:
            kwargs["max_completion_tokens"] = self._max_tokens
        else:
            kwargs["max_tokens"] = self._max_tokens
            kwargs["temperature"] = 0.2  # deterministic-ish for trading

        if require_json:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - SDK exceptions vary
            self._logger.warning("Azure call failed: %s", exc)
            raise AzureUnavailable(str(exc)) from exc
        latency_ms = int((time.monotonic() - start) * 1000)

        text = ""
        if resp.choices:
            content = resp.choices[0].message.content
            text = content or ""
        parsed: Any | None = None
        if require_json and text:
            parsed = self._safe_json_loads(text)
        return AiCallResult(text=text, parsed_json=parsed, latency_ms=latency_ms)

    @staticmethod
    def _safe_json_loads(text: str) -> Any | None:
        try:
            return json.loads(text)
        except ValueError:
            # Some models wrap JSON in ```json ... ``` fences. Strip them.
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.strip("`")
                if stripped.startswith("json"):
                    stripped = stripped[4:]
            try:
                return json.loads(stripped)
            except ValueError:
                return None
