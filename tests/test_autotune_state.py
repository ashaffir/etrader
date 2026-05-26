"""Unit tests for the rolling :class:`AutotuneState` and apply path.

We use an in-memory :class:`ConfigStore` (``":memory:"``) so we can
verify the persistence side-effect without touching disk.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field as dc_field
from typing import Any

from src.config import (
    AiConfig,
    AlertingConfig,
    AppConfig,
    AzureCredentials,
    ControlServiceConfig,
    EtoroCredentials,
    FundamentalsConfig,
    GuardrailsConfig,
    LoggingConfig,
    NewsConfig,
    OperationsConfig,
    StrategyConfig,
    ToolsConfig,
    UniverseConfig,
)
from src.config_store import ConfigStore
from src.strategy.autotune import AutotuneState
from src.strategy.autotune_parse import parse_tune_request
from src.strategy.autotune_types import TuneChange, TuneRequest


def _make_app_config(strategy: StrategyConfig, tools: ToolsConfig) -> AppConfig:
    """Build a minimal :class:`AppConfig` for autotune-state tests.

    All sections except ``strategy`` and ``tools`` are stubbed with
    defaults; the tuner only reads + writes those two.
    """
    return AppConfig(
        trading_mode="paper",
        guardrails=GuardrailsConfig(),
        operations=OperationsConfig(),
        universe=UniverseConfig(),
        news=NewsConfig(),
        fundamentals=FundamentalsConfig(),
        strategy=strategy,
        ai=AiConfig(),
        tools=tools,
        logging=LoggingConfig(),
        etoro=EtoroCredentials(
            public_key="pk", user_key="uk", is_real=False, allow_real=False,
        ),
        azure=AzureCredentials(endpoint=None, api_key=None, deployment=None),
        control=ControlServiceConfig(),
        alerting=AlertingConfig(),
    )


class AutotuneStateApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ConfigStore(":memory:")
        self.state = AutotuneState(config_store=self.store)
        self.cfg = _make_app_config(
            strategy=StrategyConfig(min_signal_strength=0.40),
            tools=ToolsConfig(spread_max_pct=0.5),
        )

    def tearDown(self) -> None:
        self.store.close()

    def test_apply_no_op_request_returns_empty(self) -> None:
        applied = self.state.apply(TuneRequest(), cfg=self.cfg)
        self.assertEqual(applied, [])

    def test_apply_updates_in_memory_field(self) -> None:
        req = parse_tune_request({
            "reason": "loosen",
            "changes": [
                {"section": "strategy", "field": "min_signal_strength",
                 "value": 0.25, "rationale": "drought"},
            ],
        })
        applied = self.state.apply(req, cfg=self.cfg)
        self.assertEqual(len(applied), 1)
        self.assertAlmostEqual(self.cfg.strategy.min_signal_strength, 0.25)
        self.assertAlmostEqual(applied[0].previous, 0.40)
        self.assertAlmostEqual(applied[0].current, 0.25)

    def test_apply_persists_to_store(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "strategy", "field": "min_signal_strength",
                 "value": 0.30, "rationale": "x"},
                {"section": "tools", "field": "spread_max_pct",
                 "value": 1.0, "rationale": "y"},
            ],
        })
        self.state.apply(req, cfg=self.cfg)
        strat_section = self.store.get_section("strategy")
        tools_section = self.store.get_section("tools")
        self.assertAlmostEqual(strat_section["min_signal_strength"], 0.30)
        self.assertAlmostEqual(tools_section["spread_max_pct"], 1.0)

    def test_apply_drops_same_value_no_op(self) -> None:
        # Current value is 0.40; LLM proposes 0.40 — should be a no-op.
        req = TuneRequest(changes=(TuneChange(
            section="strategy", field="min_signal_strength", value=0.40,
        ),))
        applied = self.state.apply(req, cfg=self.cfg)
        self.assertEqual(applied, [])

    def test_apply_int_field_coerced_correctly(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "strategy", "field": "rsi_period", "value": 21.0},
            ],
        })
        self.state.apply(req, cfg=self.cfg)
        self.assertEqual(self.cfg.strategy.rsi_period, 21)
        self.assertIsInstance(self.cfg.strategy.rsi_period, int)

    def test_apply_logs_tuning_to_internal_history(self) -> None:
        req = parse_tune_request({
            "reason": "loosen entry",
            "changes": [
                {"section": "strategy", "field": "min_signal_strength",
                 "value": 0.30, "rationale": "evidence"},
            ],
        })
        self.state.apply(req, cfg=self.cfg)
        payload = self.state.to_dict()
        self.assertEqual(len(payload["tunings"]), 1)
        entry = payload["tunings"][0]
        self.assertEqual(entry["reason"], "loosen entry")
        self.assertEqual(len(entry["changes"]), 1)


class AutotuneStateObserveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = AutotuneState(config_store=None)
        self.cfg = _make_app_config(StrategyConfig(), ToolsConfig())

    def test_drought_counter_resets_on_candidate(self) -> None:
        for cycle in range(3):
            self.state.observe_cycle(
                cycle_index=cycle, tracked_count=25,
                raw_scores=[0.1, 0.2, 0.15], candidates_count=0,
            )
        self.assertEqual(self.state.to_dict()["cycles_since_last_candidate"], 3)
        self.state.observe_cycle(
            cycle_index=3, tracked_count=25,
            raw_scores=[0.5, 0.2], candidates_count=1,
        )
        self.assertEqual(self.state.to_dict()["cycles_since_last_candidate"], 0)

    def test_record_trades_resets_trade_drought(self) -> None:
        self.state.observe_cycle(
            cycle_index=1, tracked_count=25,
            raw_scores=[0.5], candidates_count=1,
        )
        self.assertEqual(self.state.to_dict()["cycles_since_last_trade"], 1)
        self.state.record_trades_placed(trades_placed=2)
        self.assertEqual(self.state.to_dict()["cycles_since_last_trade"], 0)

    def test_evidence_carries_distribution_and_thresholds(self) -> None:
        self.state.observe_cycle(
            cycle_index=1, tracked_count=25,
            raw_scores=[0.05, 0.10, 0.18, 0.30, 0.35],
            candidates_count=0,
        )
        ev = self.state.build_evidence(
            cfg=self.cfg,
            recent_realized_pnl=[],
            open_position_pnl_total=0.0,
        )
        d = ev.to_dict()
        self.assertEqual(d["candidates_this_cycle"], 0)
        self.assertGreaterEqual(d["drought"]["cycles_since_last_candidate"], 1)
        dist = d["raw_score_distribution"]
        self.assertIn("this_cycle.max", dist)
        self.assertAlmostEqual(dist["this_cycle.max"], 0.35)
        self.assertIn("rolling.max", dist)
        self.assertAlmostEqual(
            d["current_thresholds"]["min_signal_strength"],
            self.cfg.strategy.min_signal_strength,
        )

    def test_restore_round_trips(self) -> None:
        self.state.observe_cycle(
            cycle_index=1, tracked_count=25, raw_scores=[0.1],
            candidates_count=0,
        )
        self.state.observe_cycle(
            cycle_index=2, tracked_count=25, raw_scores=[0.1],
            candidates_count=0,
        )
        payload = self.state.to_dict()
        fresh = AutotuneState(config_store=None)
        fresh.restore(payload)
        self.assertEqual(
            fresh.to_dict()["cycles_since_last_candidate"],
            payload["cycles_since_last_candidate"],
        )


if __name__ == "__main__":
    unittest.main()
