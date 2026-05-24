"""Config loader tests — both the .env parser and the TOML schema."""

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from src.config import load_config, load_env_file, summarize_config
from src.config_store import ConfigStore


class EnvParserTests(unittest.TestCase):
    def test_parses_basic_kv(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("FOO=bar\nBAZ=qux\n")
            path = Path(f.name)
        try:
            self.assertEqual(load_env_file(path), {"FOO": "bar", "BAZ": "qux"})
        finally:
            path.unlink()

    def test_skips_comments_blanks_and_strips_quotes(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("# a comment\n\nFOO=\"hello\"\nBAR='world'\nBROKEN\nexport BAZ=qux\n")
            path = Path(f.name)
        try:
            out = load_env_file(path)
            self.assertEqual(out["FOO"], "hello")
            self.assertEqual(out["BAR"], "world")
            self.assertEqual(out["BAZ"], "qux")
            self.assertNotIn("BROKEN", out)
        finally:
            path.unlink()

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_env_file(Path("/no/such/path/.env")), {})


class LoadConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        (self.root / "data").mkdir()
        # Snapshot env so we can scrub it after the test.
        self._saved_env = {
            k: os.environ.get(k) for k in (
                "PUBLIC_KEY", "PRIVATE_KEY", "REAL_USER_KEY", "ALLOW_REAL",
                "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_IS_REASONING_MODEL",
            )
        }
        for k in self._saved_env:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _write_env(self, body: str) -> Path:
        path = self.root / ".env"
        path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
        return path

    def _write_config(self, body: str) -> Path:
        path = self.root / "config.toml"
        path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
        return path

    def test_paper_mode_default(self) -> None:
        self._write_env("""
            PUBLIC_KEY=abc123
            PRIVATE_KEY=demoKey
        """)
        self._write_config("""
            [mode]
            trading = "paper"
        """)
        cfg = load_config(project_root=self.root)
        self.assertEqual(cfg.trading_mode, "paper")
        self.assertEqual(cfg.env_segment, "demo")
        self.assertEqual(cfg.etoro.user_key, "demoKey")
        self.assertFalse(cfg.etoro.is_real)
        # Defaults applied
        self.assertEqual(cfg.guardrails.max_per_trade_usd, 500.0)
        self.assertEqual(cfg.operations.check_interval_seconds, 60)

    def test_live_mode_requires_allow_real_and_real_key(self) -> None:
        self._write_env("""
            PUBLIC_KEY=abc123
            PRIVATE_KEY=demoKey
        """)
        self._write_config("""
            [mode]
            trading = "live"
        """)
        with self.assertRaises(RuntimeError) as cm:
            load_config(project_root=self.root)
        self.assertIn("ALLOW_REAL", str(cm.exception))

    def test_live_mode_works_when_gated(self) -> None:
        self._write_env("""
            PUBLIC_KEY=abc123
            PRIVATE_KEY=demoKey
            REAL_USER_KEY=realKey
            ALLOW_REAL=true
        """)
        self._write_config("""
            [mode]
            trading = "live"
        """)
        cfg = load_config(project_root=self.root)
        self.assertTrue(cfg.etoro.is_real)
        self.assertEqual(cfg.env_segment, "real")
        self.assertEqual(cfg.etoro.user_key, "realKey")

    def test_invalid_mode_rejected(self) -> None:
        self._write_env("PUBLIC_KEY=a\nPRIVATE_KEY=b\n")
        self._write_config("""
            [mode]
            trading = "yolo"
        """)
        with self.assertRaises(ValueError):
            load_config(project_root=self.root)

    def test_missing_public_key_rejected(self) -> None:
        self._write_env("PRIVATE_KEY=demoOnly\n")
        self._write_config("""
            [mode]
            trading = "paper"
        """)
        with self.assertRaises(RuntimeError):
            load_config(project_root=self.root)

    def test_first_run_snapshots_effective_config_to_db(self) -> None:
        self._write_env("""
            PUBLIC_KEY=abc
            PRIVATE_KEY=demoKey
        """)
        self._write_config("""
            [mode]
            trading = "paper"

            [guardrails]
            max_per_trade_usd = 750.0
        """)
        load_config(project_root=self.root)
        store = ConfigStore(self.root / "data" / "config.sqlite")
        try:
            guardrails = store.get_section("guardrails")
            # Snapshot must contain BOTH the TOML override and the dataclass defaults.
            self.assertEqual(guardrails["max_per_trade_usd"], 750.0)
            self.assertEqual(guardrails["max_parallel_trades"], 10)
            # Every persisted section gets at least one row.
            self.assertNotEqual(store.get_section("operations"), {})
            self.assertNotEqual(store.get_section("strategy"), {})
        finally:
            store.close()

    def test_db_overrides_beat_toml_on_subsequent_load(self) -> None:
        """User edits a guardrail at runtime → restart sees the new value, not TOML."""
        self._write_env("""
            PUBLIC_KEY=abc
            PRIVATE_KEY=demoKey
        """)
        self._write_config("""
            [mode]
            trading = "paper"

            [guardrails]
            max_per_trade_usd      = 500.0
            default_stop_loss_pct  = 5.0
            default_take_profit_pct = 8.0
        """)

        # Boot once → DB gets snapshotted with TOML values.
        cfg1 = load_config(project_root=self.root)
        self.assertEqual(cfg1.guardrails.default_stop_loss_pct, 5.0)

        # Simulate operator editing SL via Telegram.
        store = ConfigStore(self.root / "data" / "config.sqlite")
        try:
            store.set_field("guardrails", "default_stop_loss_pct", 7.0)
            store.set_field("guardrails", "default_take_profit_pct", 12.0)
        finally:
            store.close()

        # Reboot — DB-stored overrides must win.
        cfg2 = load_config(project_root=self.root)
        self.assertEqual(cfg2.guardrails.default_stop_loss_pct, 7.0)
        self.assertEqual(cfg2.guardrails.default_take_profit_pct, 12.0)
        # Fields not changed at runtime still come through unchanged.
        self.assertEqual(cfg2.guardrails.max_per_trade_usd, 500.0)

    def test_db_wins_even_when_toml_is_edited_between_restarts(self) -> None:
        """Operator changed TP via Telegram; later edits config.toml. DB wins."""
        self._write_env("""
            PUBLIC_KEY=abc
            PRIVATE_KEY=demoKey
        """)
        self._write_config("""
            [mode]
            trading = "paper"

            [guardrails]
            default_take_profit_pct = 8.0
        """)
        load_config(project_root=self.root)

        store = ConfigStore(self.root / "data" / "config.sqlite")
        try:
            store.set_field("guardrails", "default_take_profit_pct", 12.0)
        finally:
            store.close()

        # Now someone edits config.toml — bumps the TOML default to 9.
        self._write_config("""
            [mode]
            trading = "paper"

            [guardrails]
            default_take_profit_pct = 9.0
        """)

        cfg = load_config(project_root=self.root)
        # DB value (12.0) wins, NOT the new TOML default (9.0).
        self.assertEqual(cfg.guardrails.default_take_profit_pct, 12.0)

    def test_unknown_db_keys_are_ignored(self) -> None:
        """Schema evolution: removed-field DB rows must not break startup."""
        self._write_env("""
            PUBLIC_KEY=abc
            PRIVATE_KEY=demoKey
        """)
        self._write_config("""
            [mode]
            trading = "paper"
        """)
        # Seed DB with a bogus field before first load.
        (self.root / "data").mkdir(exist_ok=True)
        store = ConfigStore(self.root / "data" / "config.sqlite")
        try:
            store.set_field("guardrails", "fictional_legacy_key", "ignored")
            store.set_field("guardrails", "max_per_trade_usd", 200.0)
        finally:
            store.close()

        cfg = load_config(project_root=self.root)
        self.assertEqual(cfg.guardrails.max_per_trade_usd, 200.0)

    def test_summary_omits_secrets(self) -> None:
        self._write_env("""
            PUBLIC_KEY=verysecret
            PRIVATE_KEY=demoKey
        """)
        self._write_config("""
            [mode]
            trading = "paper"
        """)
        cfg = load_config(project_root=self.root)
        summary = summarize_config(cfg)
        self.assertNotIn("verysecret", summary)
        self.assertNotIn("demoKey", summary)
        self.assertIn("mode=paper", summary)


if __name__ == "__main__":
    unittest.main()
