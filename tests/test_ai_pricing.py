"""Tests for src/ai/pricing.py."""

from __future__ import annotations

import unittest

from src.ai.pricing import TokenRates, known_families, lookup_rates, price_table_as_of


class LookupTests(unittest.TestCase):
    def test_gpt5_exact_match(self) -> None:
        rates = lookup_rates("gpt-5")
        self.assertIsNotNone(rates)
        assert rates is not None
        self.assertEqual(rates.family, "gpt-5")
        self.assertEqual(rates.input_per_m, 1.25)
        self.assertEqual(rates.output_per_m, 10.00)
        self.assertEqual(rates.cached_per_m, 0.13)

    def test_gpt5_prefix_match_with_versioned_deployment(self) -> None:
        # Real-world deployment names look like "gpt-5.4-2025-08-07"
        # — the table prefix-matches on "gpt-5" so we still get the
        # flagship rates.
        rates = lookup_rates("gpt-5.4-2025-08-07")
        self.assertIsNotNone(rates)
        assert rates is not None
        self.assertEqual(rates.family, "gpt-5")

    def test_mini_takes_precedence_over_flagship(self) -> None:
        rates = lookup_rates("gpt-5-mini-prod")
        self.assertIsNotNone(rates)
        assert rates is not None
        self.assertEqual(rates.family, "gpt-5-mini")
        self.assertEqual(rates.input_per_m, 0.25)

    def test_nano_takes_precedence_over_mini(self) -> None:
        rates = lookup_rates("gpt-5-nano-eu")
        self.assertIsNotNone(rates)
        assert rates is not None
        self.assertEqual(rates.family, "gpt-5-nano")

    def test_case_insensitive(self) -> None:
        a = lookup_rates("GPT-5")
        b = lookup_rates("gpt-5")
        self.assertEqual(a, b)

    def test_unknown_deployment_returns_none(self) -> None:
        self.assertIsNone(lookup_rates("acme-llm-xl"))
        self.assertIsNone(lookup_rates(""))
        self.assertIsNone(lookup_rates(None))


class CostMathTests(unittest.TestCase):
    rates = TokenRates("test", input_per_m=1.0, cached_per_m=0.1, output_per_m=4.0)

    def test_pure_prompt_and_completion(self) -> None:
        cost = self.rates.cost_for(prompt_tokens=1_000_000, completion_tokens=500_000)
        # 1M @ $1 + 500k @ $4 = $1 + $2 = $3
        self.assertAlmostEqual(cost, 3.0, places=6)

    def test_cached_tokens_pay_discount_rate(self) -> None:
        cost = self.rates.cost_for(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=500_000,
        )
        # 500k @ $1 + 500k @ $0.10 = $0.50 + $0.05 = $0.55
        self.assertAlmostEqual(cost, 0.55, places=6)

    def test_cached_capped_at_prompt(self) -> None:
        cost = self.rates.cost_for(
            prompt_tokens=200_000,
            completion_tokens=0,
            cached_tokens=999_999,
        )
        # cached must not exceed prompt → 200k @ $0.10 = $0.02
        self.assertAlmostEqual(cost, 0.02, places=6)

    def test_no_cached_rate_treats_cached_as_standard(self) -> None:
        rates = TokenRates(
            "no-cache", input_per_m=1.0, cached_per_m=None, output_per_m=4.0,
        )
        cost = rates.cost_for(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=500_000,
        )
        # Without cached rate, the whole 1M counts as standard input.
        self.assertAlmostEqual(cost, 1.0, places=6)

    def test_negative_clamped_to_zero(self) -> None:
        cost = self.rates.cost_for(
            prompt_tokens=-5, completion_tokens=-5, cached_tokens=-5,
        )
        self.assertEqual(cost, 0.0)


class MetadataTests(unittest.TestCase):
    def test_price_table_as_of_is_isolated(self) -> None:
        as_of = price_table_as_of()
        self.assertRegex(as_of, r"^\d{4}-\d{2}$")

    def test_known_families_includes_flagship(self) -> None:
        families = list(known_families())
        self.assertIn("gpt-5", families)
        self.assertIn("gpt-5-mini", families)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
