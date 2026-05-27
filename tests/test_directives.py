"""Tests for src/strategy/directives.py."""

from __future__ import annotations

import unittest

from src.strategy.directives import (
    DirectiveError,
    Directives,
    DirectivesStore,
    NOTES_MAX_CHARS,
    STRUCTURED_KEYS,
    coerce_value,
)


class DirectivesDataclassTests(unittest.TestCase):
    def test_defaults_are_disabled(self) -> None:
        d = Directives()
        self.assertFalse(d.no_overnight)
        self.assertEqual(d.hold_ceiling_minutes, 0)
        self.assertEqual(d.blocked_symbols, ())
        self.assertEqual(d.blocked_sectors, ())
        self.assertEqual(d.max_total_account_invested_usd, 0.0)
        self.assertEqual(d.notes, "")
        self.assertFalse(d.is_symbol_blocked("AAPL"))
        self.assertFalse(d.is_sector_blocked("Technology"))

    def test_to_from_dict_round_trip(self) -> None:
        d = Directives(
            no_overnight=True,
            hold_ceiling_minutes=120,
            blocked_symbols=("NVDA", "TSLA"),
            blocked_sectors=("Energy",),
            max_total_account_invested_usd=3000.0,
            notes="prefer financials",
        )
        round_tripped = Directives.from_dict(d.to_dict())
        self.assertEqual(round_tripped, d)

    def test_is_symbol_blocked_is_case_insensitive(self) -> None:
        d = Directives(blocked_symbols=("NVDA",))
        self.assertTrue(d.is_symbol_blocked("nvda"))
        self.assertTrue(d.is_symbol_blocked(" Nvda "))
        self.assertFalse(d.is_symbol_blocked("AMD"))

    def test_is_sector_blocked_is_case_insensitive(self) -> None:
        d = Directives(blocked_sectors=("Energy",))
        self.assertTrue(d.is_sector_blocked("energy"))
        self.assertTrue(d.is_sector_blocked(" ENERGY "))
        self.assertFalse(d.is_sector_blocked("Technology"))
        self.assertFalse(d.is_sector_blocked(None))


class CoerceValueTests(unittest.TestCase):
    def test_unknown_key_raises(self) -> None:
        with self.assertRaises(DirectiveError):
            coerce_value("does_not_exist", True)

    def test_bool_coercion(self) -> None:
        self.assertTrue(coerce_value("no_overnight", "true"))
        self.assertTrue(coerce_value("no_overnight", "YES"))
        self.assertTrue(coerce_value("no_overnight", 1))
        self.assertFalse(coerce_value("no_overnight", "no"))
        self.assertFalse(coerce_value("no_overnight", "off"))
        with self.assertRaises(DirectiveError):
            coerce_value("no_overnight", "maybe")

    def test_int_coercion_rejects_negative(self) -> None:
        self.assertEqual(coerce_value("hold_ceiling_minutes", "60"), 60)
        self.assertEqual(coerce_value("hold_ceiling_minutes", 60.0), 60)
        with self.assertRaises(DirectiveError):
            coerce_value("hold_ceiling_minutes", -5)
        with self.assertRaises(DirectiveError):
            coerce_value("hold_ceiling_minutes", "junk")

    def test_float_coercion_rejects_negative(self) -> None:
        self.assertEqual(coerce_value("max_total_account_invested_usd", "1500"), 1500.0)
        self.assertEqual(coerce_value("max_total_account_invested_usd", 0), 0.0)
        with self.assertRaises(DirectiveError):
            coerce_value("max_total_account_invested_usd", -10)

    def test_symbol_list_dedupe_and_uppercase(self) -> None:
        result = coerce_value("blocked_symbols", "nvda, TSLA, NVDA, brk.b")
        self.assertEqual(result, ("NVDA", "TSLA", "BRK.B"))

    def test_symbol_list_rejects_invalid_chars(self) -> None:
        with self.assertRaises(DirectiveError):
            coerce_value("blocked_symbols", "NVDA, t$la")

    def test_label_list_preserves_case_but_dedupes(self) -> None:
        result = coerce_value("blocked_sectors", "Energy, energy, Healthcare")
        # First entry retained verbatim; case-insensitive dedupe drops the second.
        self.assertEqual(result, ("Energy", "Healthcare"))


class DirectivesStoreTests(unittest.TestCase):
    def test_set_and_clear_persist_to_dataclass(self) -> None:
        store = DirectivesStore()
        prev, cur = store.set_field("no_overnight", "true")
        self.assertFalse(prev)
        self.assertTrue(cur)
        self.assertTrue(store.current().no_overnight)
        prev, cur = store.clear_field("no_overnight")
        self.assertTrue(prev)
        self.assertFalse(cur)
        self.assertFalse(store.current().no_overnight)

    def test_set_unknown_key_raises(self) -> None:
        store = DirectivesStore()
        with self.assertRaises(DirectiveError):
            store.set_field("nope", "x")
        with self.assertRaises(DirectiveError):
            store.clear_field("nope")

    def test_set_blocked_symbols_round_trip(self) -> None:
        store = DirectivesStore()
        prev, cur = store.set_field("blocked_symbols", "NVDA, TSLA")
        self.assertEqual(prev, ())
        self.assertEqual(cur, ("NVDA", "TSLA"))
        self.assertTrue(store.current().is_symbol_blocked("NVDA"))

    def test_notes_cap_enforced(self) -> None:
        store = DirectivesStore()
        long = "x" * (NOTES_MAX_CHARS + 50)
        prev, cur = store.set_notes(long)
        self.assertEqual(prev, "")
        self.assertEqual(len(cur), NOTES_MAX_CHARS)

    def test_notes_clear(self) -> None:
        store = DirectivesStore()
        store.set_notes("hello")
        previous = store.clear_notes()
        self.assertEqual(previous, "hello")
        self.assertEqual(store.current().notes, "")

    def test_persistence_round_trip(self) -> None:
        store = DirectivesStore()
        store.set_field("no_overnight", "true")
        store.set_field("blocked_symbols", "NVDA")
        store.set_notes("prefer financials")
        payload = store.to_persistable()

        clone = DirectivesStore()
        clone.restore(payload)
        cur = clone.current()
        self.assertTrue(cur.no_overnight)
        self.assertEqual(cur.blocked_symbols, ("NVDA",))
        self.assertEqual(cur.notes, "prefer financials")

    def test_structured_keys_covers_all_editable_fields(self) -> None:
        # When a field gets added to ``Directives`` (except ``notes``,
        # which has its own editor) it MUST also be added to
        # STRUCTURED_KEYS so the Telegram + HTTP layer can edit it.
        editable = {f for f in Directives.__dataclass_fields__ if f != "notes"}
        self.assertEqual(set(STRUCTURED_KEYS), editable)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
