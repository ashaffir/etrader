"""Parse + coerce the LLM's ``tuning`` JSON block into a :class:`TuneRequest`.

Defensive on purpose: the LLM is allowed to return whatever, but the
bot must never crash on a malformed suggestion. Unknown sections /
fields are dropped silently (logged at DEBUG by the caller); type
coercion failures drop just that one change.

This module is import-light so test code can hammer it without pulling
in the rest of the strategy stack.
"""

from __future__ import annotations

from typing import Any, Iterable

from .autotune_types import (
    ALLOWED_SECTIONS,
    STRATEGY_FIELDS,
    TOOLS_FIELDS,
    TuneChange,
    TuneRequest,
    field_kind,
)


def _section_field_names(section: str) -> tuple[str, ...]:
    if section == "strategy":
        return STRATEGY_FIELDS
    if section == "tools":
        return TOOLS_FIELDS
    return ()


def _coerce_value(section: str, field_name: str, raw: Any) -> Any | None:
    """Coerce ``raw`` to the field's declared type. Return None on failure."""
    kind = field_kind(section, field_name)
    if kind == "unknown":
        return None
    try:
        if kind == "int":
            return int(float(raw))
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_tune_request(payload: Any) -> TuneRequest:
    """Build a :class:`TuneRequest` from the LLM's parsed JSON.

    ``payload`` is the raw value of the top-level ``tuning`` key. The
    expected shape is::

        {"changes": [{"section", "field", "value", "rationale"}, ...],
         "reason": <string>}

    Anything else is reduced to an empty request. We do NOT raise:
    the LLM's tuning block is *advisory* and a malformed one must not
    abort the cycle's main work (the trade decisions).
    """
    if not isinstance(payload, dict):
        return TuneRequest()

    reason = str(payload.get("reason") or "").strip()
    raw_changes: Any = payload.get("changes") or []
    if not isinstance(raw_changes, list):
        return TuneRequest(reason=reason)

    parsed: list[TuneChange] = []
    seen: set[tuple[str, str]] = set()
    for entry in raw_changes:
        change = _parse_change(entry)
        if change is None:
            continue
        # Dedup by (section, field) — keep the FIRST occurrence so the
        # LLM can't accidentally double-tap the same field with two
        # different values in one cycle.
        key = (change.section, change.field)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(change)

    return TuneRequest(changes=tuple(parsed), reason=reason)


def _parse_change(entry: Any) -> TuneChange | None:
    if not isinstance(entry, dict):
        return None
    section = str(entry.get("section") or "").strip().lower()
    if section not in ALLOWED_SECTIONS:
        return None
    field_name = str(entry.get("field") or "").strip()
    if field_name not in _section_field_names(section):
        return None
    coerced = _coerce_value(section, field_name, entry.get("value"))
    if coerced is None:
        return None
    rationale = str(entry.get("rationale") or "").strip()
    return TuneChange(
        section=section,
        field=field_name,
        value=coerced,
        rationale=rationale,
    )


def render_tune_diff(applied: Iterable[Any]) -> str:
    """Render an applied-tuning batch as a single-line human diff.

    Used by the Telegram alert and the trader log; takes an iterable
    of :class:`TuneApplied` (typed as ``Any`` to avoid a circular import).
    """
    parts: list[str] = []
    for a in applied:
        prev = getattr(a, "previous", None)
        cur = getattr(a, "current", None)
        sec = getattr(a, "section", "?")
        fld = getattr(a, "field", "?")
        parts.append(f"{sec}.{fld}: {prev} → {cur}")
    return "; ".join(parts) if parts else "(no-op)"
