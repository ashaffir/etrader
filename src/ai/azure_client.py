"""Azure AI Foundry / Azure OpenAI chat-completions wrapper.

Imports are lazy: the bot can run without ``openai`` installed if AI
features are disabled (``[ai] enabled = false`` in config). The wrapper
also tolerates reasoning-model deployments (gpt-5 / o-series) by
omitting ``temperature`` and using ``max_completion_tokens``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import AzureCredentials
from .usage_tracker import LLMUsageTracker


class AzureUnavailable(Exception):
    """The ``openai`` package isn't installed or credentials are missing."""


@dataclass(frozen=True)
class AiCallResult:
    text: str
    parsed_json: Any | None
    latency_ms: int
    # Token accounting populated from the SDK ``usage`` block when
    # the upstream response carries one. Set to 0 / None when the
    # SDK doesn't report usage (e.g. error paths) so downstream
    # accounting silently skips that call instead of attributing
    # spurious totals.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


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
        usage_tracker: LLMUsageTracker | None = None,
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
        self._usage_tracker = usage_tracker
        self._client = AzureOpenAI(
            azure_endpoint=credentials.endpoint,
            api_key=credentials.api_key,
            api_version=credentials.api_version,
        )

    def set_usage_tracker(self, tracker: LLMUsageTracker | None) -> None:
        """Attach a usage tracker after construction.

        Useful for tests and for the wiring path where the bot
        constructs the client first (to verify creds + import the
        SDK) and only later builds the tracker once it knows the
        deployment name.
        """
        self._usage_tracker = tracker

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        require_json: bool = True,
        call_type: str = "unknown",
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
        prompt_tokens, completion_tokens, cached_tokens, total_tokens = (
            self._extract_usage(resp)
        )
        if self._usage_tracker is not None and total_tokens > 0:
            try:
                self._usage_tracker.record(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    call_type=call_type,
                    latency_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001 - tracking must never block trading
                self._logger.warning("usage tracker failed: %s", exc)
        return AiCallResult(
            text=text,
            parsed_json=parsed,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _extract_usage(resp: Any) -> tuple[int, int, int, int]:
        """Pull ``(prompt, completion, cached, total)`` out of the SDK response.

        OpenAI's Python SDK exposes ``resp.usage`` with
        ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``;
        cached counts live under ``prompt_tokens_details.cached_tokens``
        on the newer responses. Older shapes (or mocked responses in
        tests) may have ``usage`` as None or a dict — we tolerate both.
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            return 0, 0, 0, 0
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            total = int(usage.get("total_tokens") or (prompt + completion))
            details = usage.get("prompt_tokens_details") or {}
            cached = int((details or {}).get("cached_tokens") or 0)
            return prompt, completion, cached, total
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        details = getattr(usage, "prompt_tokens_details", None)
        cached = 0
        if details is not None:
            if isinstance(details, dict):
                cached = int(details.get("cached_tokens") or 0)
            else:
                cached = int(getattr(details, "cached_tokens", 0) or 0)
        return prompt, completion, cached, total

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
