"""InstrumentFeedTool — surface eToro's social discussion volume + tilt.

The fetcher (a duck-typed object exposing ``.fetch(instrument_id)``)
is injected at registry construction so unit tests can use a stub
without needing the eToro client. When the fetcher is ``None``, the
tool quietly no-ops.
"""

from __future__ import annotations

from typing import Any

from .base import AssetClass, Tool, ToolContext, ToolResult


class InstrumentFeedTool(Tool):
    name = "instrument_feed"
    family = "context"
    role = "feature"
    purpose = "eToro newsfeed activity + bullish/bearish keyword tilt"
    asset_classes = (
        AssetClass.STOCK, AssetClass.CRYPTO, AssetClass.ETF,
        AssetClass.INDEX, AssetClass.COMMODITY, AssetClass.FX, AssetClass.OTHER,
    )

    def __init__(self, *, fetcher: Any | None = None) -> None:
        self._fetcher = fetcher

    def applies_to(self, ctx: ToolContext) -> bool:
        return self._fetcher is not None and super().applies_to(ctx)

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        if self._fetcher is None:
            return ToolResult(features={"feed": None}, score=0.0)
        try:
            summary = self._fetcher.fetch(ctx.instrument_id)
        except Exception:  # noqa: BLE001 - feed is never trade-critical
            return ToolResult(features={"feed_error": True}, score=0.0)
        bullish = int(getattr(summary, "bullish_keyword_count", 0))
        bearish = int(getattr(summary, "bearish_keyword_count", 0))
        posts_24h = int(getattr(summary, "posts_24h", 0))
        score = 0.0
        if bullish + bearish > 0:
            tilt = (bullish - bearish) / float(bullish + bearish)
            if ctx.candidate_action == "BUY":
                score = max(-0.3, min(0.3, tilt))
            else:
                score = max(-0.3, min(0.3, -tilt))
        features = {
            "post_count": int(getattr(summary, "post_count", 0)),
            "posts_24h": posts_24h,
            "bullish_keywords": bullish,
            "bearish_keywords": bearish,
        }
        sample_titles = getattr(summary, "sample_titles", None)
        if sample_titles:
            features["sample_titles"] = list(sample_titles)
        return ToolResult(features=features, score=score)
