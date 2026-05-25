"""Live "dry-run" probe for individual news sources.

The aggregator runs every source on a schedule and folds the results
into the candidate store; that's enough for the regular trading loop
but it doesn't help an operator who wants to answer *right now*:

    "Is StockTwits actually reachable?  Is SEC EDGAR returning rows?
     Is my Yahoo RSS proxy still working?"

This module exposes a single :func:`probe_source` helper that runs one
source's :meth:`fetch` in isolation, captures any error / latency /
sample item, and returns a structured :class:`ChannelProbeResult`. The
controller exposes it via the ``/news/channels/test`` endpoint and the
Telegram ``/channels test`` command.

Probes are deliberately *side-effect-free*: results are NOT folded
into the candidate store. The probe is a health check, not a back-door
ingest path — so an operator can run it as often as they want without
double-counting observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .sources.base import NewsItem, NewsSource


@dataclass(frozen=True)
class ChannelProbeResult:
    """Outcome of one ``source.fetch()`` invocation.

    Attributes
    ----------
    name:
        Canonical source name (``stocktwits``, ``sec_edgar``, …).
    ok:
        True when the source returned *any* iterable of items without
        raising (even an empty list counts — a healthy SEC EDGAR feed
        legitimately returns 0 new 8-Ks for a few minutes at a time).
    items_count:
        Number of :class:`NewsItem` objects the source emitted.
    sample_headline:
        First headline string (truncated for readability), useful as a
        sanity-check that the response actually parsed.
    duration_ms:
        Wall-clock latency of the fetch call.
    error:
        Stringified exception when ``ok=False``; ``None`` otherwise.
    disabled_reason:
        When the source self-reports as disabled (e.g. SEC EDGAR with
        no ``SEC_USER_AGENT``), the probe captures the reason without
        actually invoking ``fetch``.
    """

    name: str
    ok: bool
    items_count: int = 0
    sample_headline: str | None = None
    duration_ms: int = 0
    error: str | None = None
    disabled_reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "items_count": int(self.items_count),
            "sample_headline": self.sample_headline,
            "duration_ms": int(self.duration_ms),
            "error": self.error,
            "disabled_reason": self.disabled_reason,
            "metadata": dict(self.metadata),
        }


def probe_source(
    source: NewsSource,
    *,
    known_symbols: Iterable[str] | None = None,
    headline_max_len: int = 90,
) -> ChannelProbeResult:
    """Run ``source.fetch`` once and return a structured result.

    Never raises: any exception thrown by the source is captured into
    :attr:`ChannelProbeResult.error`. ``disabled_reason`` is surfaced
    when the source carries a ``_disabled_reason`` attribute (used by
    :class:`~src.news.sources.sec_edgar.SecEdgar8KSource`).
    """
    name = str(getattr(source, "name", source.__class__.__name__))
    disabled = _disabled_reason(source)
    if disabled:
        return ChannelProbeResult(
            name=name,
            ok=False,
            disabled_reason=disabled,
        )

    started = time.monotonic()
    try:
        items_iter = source.fetch(
            since=None,
            known_symbols=list(known_symbols) if known_symbols is not None else None,
        )
        items = list(items_iter)
    except Exception as exc:  # noqa: BLE001 — surface, never crash
        duration_ms = int((time.monotonic() - started) * 1000)
        return ChannelProbeResult(
            name=name,
            ok=False,
            duration_ms=duration_ms,
            error=f"{type(exc).__name__}: {exc}",
        )

    duration_ms = int((time.monotonic() - started) * 1000)
    items = [it for it in items if isinstance(it, NewsItem)]
    sample = items[0].headline if items else None
    if sample and len(sample) > headline_max_len:
        sample = sample[:headline_max_len - 1] + "…"
    return ChannelProbeResult(
        name=name,
        ok=True,
        items_count=len(items),
        sample_headline=sample,
        duration_ms=duration_ms,
    )


def probe_many(
    sources: Sequence[NewsSource],
    *,
    only: Iterable[str] | None = None,
    known_symbols: Iterable[str] | None = None,
) -> list[ChannelProbeResult]:
    """Probe a list of sources in order, optionally filtered by ``only``.

    ``only`` is a case-insensitive set of source names; when supplied,
    unmatched names are dropped silently so the caller can pass a
    user-provided filter without having to pre-validate it.
    """
    name_filter = (
        {n.strip().lower() for n in only if n and n.strip()}
        if only is not None
        else None
    )
    known_list = list(known_symbols) if known_symbols is not None else None
    out: list[ChannelProbeResult] = []
    for source in sources:
        name = str(getattr(source, "name", source.__class__.__name__)).lower()
        if name_filter is not None and name not in name_filter:
            continue
        out.append(probe_source(source, known_symbols=known_list))
    return out


def _disabled_reason(source: NewsSource) -> str | None:
    """Return the source-reported disabled reason if it exposes one."""
    raw = getattr(source, "_disabled_reason", None)
    if raw is None:
        return None
    reason = str(raw).strip()
    return reason or None
