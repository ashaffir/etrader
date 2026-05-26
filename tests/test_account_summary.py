"""Tests for the account-summary aggregation (per *etoro-account-snapshot* rule).

These tests pin down the four formulas — Available Cash, Total Invested,
Profit/Loss, Equity — using a synthetic ``clientPortfolio`` payload. We
test against both the eToro-spec capital-suffix casing and the legacy
lowerCamel example, since :mod:`src.etoro.trading` accepts either.
"""

import unittest

from src.etoro.trading import (
    PortfolioSnapshot,
    compute_account_summary,
    fetch_portfolio,
)


class _ClientStub:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_path: str | None = None

    def get(self, path: str, params=None, retries: int = 0):  # noqa: ARG002
        self.last_path = path
        return self.payload


class FetchPortfolioTests(unittest.TestCase):
    def test_parses_minimal_payload(self) -> None:
        payload = {
            "clientPortfolio": {
                "credit": 10_000.0,
                "unrealizedPnL": 250.0,
                "positions": [
                    {
                        "positionID": 9001,
                        "instrumentID": 101,
                        "isBuy": True,
                        "openRate": 100.0,
                        "amount": 500.0,
                        "units": 5.0,
                        "leverage": 1,
                        "mirrorID": 0,
                        "pnL": 25.0,
                    }
                ],
                "ordersForOpen": [],
                "orders": [],
                "mirrors": [],
            }
        }
        snap = fetch_portfolio(_ClientStub(payload), "demo")
        self.assertEqual(snap.credit, 10_000.0)
        self.assertEqual(len(snap.positions), 1)
        self.assertEqual(snap.positions[0].position_id, 9001)
        self.assertEqual(snap.positions[0].pnl, 25.0)


class AccountSummaryFormulaTests(unittest.TestCase):
    def test_formulas_match_etoro_spec(self) -> None:
        payload = {
            "clientPortfolio": {
                "credit": 10_000.0,
                "unrealizedPnL": 0.0,
                "positions": [
                    {
                        "positionID": 1, "instrumentID": 100, "isBuy": True,
                        "openRate": 100.0, "amount": 500.0, "units": 5.0,
                        "leverage": 1, "mirrorID": 0, "pnL": 50.0,
                    },
                    {
                        "positionID": 2, "instrumentID": 200, "isBuy": True,
                        "openRate": 50.0, "amount": 1_000.0, "units": 20.0,
                        "leverage": 1, "mirrorID": 0, "pnL": -30.0,
                    },
                ],
                "ordersForOpen": [
                    {"orderID": 5, "instrumentID": 300, "amount": 200.0,
                     "isBuy": True, "leverage": 1, "mirrorID": 0,
                     "totalExternalCosts": 1.0},
                ],
                "orders": [
                    {"orderId": 6, "instrumentId": 400, "amount": 50.0},
                ],
                "mirrors": [],
            }
        }
        snap = fetch_portfolio(_ClientStub(payload), "demo")
        s = compute_account_summary(snap)

        # Available cash = 10000 - 200 (manual orderForOpen) - 50 (limit order) = 9750
        self.assertAlmostEqual(s["available_cash"], 9_750.0, places=2)
        # Total invested = 500 + 1000 (positions) + 200 (manual orderForOpen) +
        #                  50 (limit order) + 1 (external costs)
        self.assertAlmostEqual(s["total_invested"], 1_751.0, places=2)
        # PnL = 50 + (-30) = 20
        self.assertAlmostEqual(s["profit_loss"], 20.0, places=2)
        # Equity = AC + TI + PnL
        self.assertAlmostEqual(s["equity"], 9_750.0 + 1_751.0 + 20.0, places=2)

    def test_pnl_falls_back_to_top_level_when_positions_report_zero(self) -> None:
        """Live observation: eToro returns per-position pnL=0 for many account
        compositions (e.g. unleveraged cash equities) while the rolled-up
        ``unrealizedPnL`` carries the truth. The old summary summed the
        per-position fields and silently reported $0.00 P/L when the account
        was actually down. This test pins the fix in place."""
        payload = {
            "clientPortfolio": {
                "credit": 100_000.0,
                "unrealizedPnL": -5.91,      # the real, rolled-up loss
                "positions": [
                    {"positionID": 1, "instrumentID": 100, "isBuy": True,
                     "openRate": 100.0, "amount": 500.0, "units": 5.0,
                     "leverage": 1, "mirrorID": 0, "pnL": 0.0},
                    {"positionID": 2, "instrumentID": 200, "isBuy": True,
                     "openRate": 50.0, "amount": 500.0, "units": 10.0,
                     "leverage": 1, "mirrorID": 0, "pnL": 0.0},
                ],
                "ordersForOpen": [],
                "orders": [],
                "mirrors": [],
            }
        }
        snap = fetch_portfolio(_ClientStub(payload), "demo")
        s = compute_account_summary(snap)
        self.assertAlmostEqual(s["profit_loss"], -5.91, places=2)

    def test_handles_lowercamel_payload(self) -> None:
        # Same logical state, lowerCamel keys (matches the OpenAPI example).
        payload = {
            "clientPortfolio": {
                "credit": 1_000.0,
                "unrealizedPnL": 0.0,
                "positions": [
                    {"positionId": 1, "instrumentId": 100, "isBuy": True,
                     "openRate": 100.0, "amount": 200.0, "units": 2.0,
                     "leverage": 1, "mirrorId": 0, "pnL": 12.0},
                ],
                "ordersForOpen": [],
                "orders": [],
                "mirrors": [],
            }
        }
        snap = fetch_portfolio(_ClientStub(payload), "demo")
        self.assertEqual(snap.positions[0].position_id, 1)
        self.assertEqual(snap.positions[0].pnl, 12.0)
        s = compute_account_summary(snap)
        self.assertAlmostEqual(s["available_cash"], 1_000.0, places=2)
        self.assertAlmostEqual(s["total_invested"], 200.0, places=2)
        self.assertAlmostEqual(s["profit_loss"], 12.0, places=2)


if __name__ == "__main__":
    unittest.main()
