"""``/api/v1/me`` — authenticated user identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import EtoroClient


@dataclass(frozen=True)
class IdentityInfo:
    gcid: int | None
    real_cid: int | None
    demo_cid: int | None

    @classmethod
    def from_response(cls, raw: dict[str, Any]) -> "IdentityInfo":
        return cls(
            gcid=raw.get("gcid"),
            real_cid=raw.get("realCid"),
            demo_cid=raw.get("demoCid"),
        )


def fetch_identity(client: EtoroClient) -> IdentityInfo:
    raw = client.get("/me")
    return IdentityInfo.from_response(raw or {})
