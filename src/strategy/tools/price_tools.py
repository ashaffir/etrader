"""Re-export shim for the price-tool subpackage.

The original implementation lived in this file; we split it into
``tools/price/{moving_averages,oscillators,envelopes}.py`` to keep
each module under the project's 300-line guideline. Imports of the
old names continue to work unchanged.
"""

from __future__ import annotations

from .base import Tool
from .price import (
    BollingerTool,
    DonchianTool,
    EmaCrossTool,
    MacdTool,
    RsiTool,
    SmaCrossTool,
    TrendFilterTool,
)

__all__ = [
    "BollingerTool",
    "DonchianTool",
    "EmaCrossTool",
    "MacdTool",
    "RsiTool",
    "SmaCrossTool",
    "TrendFilterTool",
    "build_tools",
]


def build_tools() -> list[Tool]:
    return [
        SmaCrossTool(),
        EmaCrossTool(),
        RsiTool(),
        MacdTool(),
        BollingerTool(),
        DonchianTool(),
        TrendFilterTool(),
    ]
