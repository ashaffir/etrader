"""Tests for the NewsConfig section wiring in config.py + config_store."""

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.config import NewsConfig, load_config
from src.config_store import PERSISTED_SECTIONS, open_store


class NewsConfigWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / ".env").write_text(
            "PUBLIC_KEY=test-pk\nPRIVATE_KEY=test-uk\n", encoding="utf-8"
        )

    def _write_config(self, body: str) -> Path:
        path = self.tmp / "config.toml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_news_section_in_persisted_sections(self) -> None:
        self.assertIn("news", PERSISTED_SECTIONS)

    def test_defaults_when_section_missing(self) -> None:
        self._write_config('[mode]\ntrading = "paper"\n')
        cfg = load_config(
            project_root=self.tmp,
            env_path=self.tmp / ".env",
            config_path=self.tmp / "config.toml",
            config_db_path=self.tmp / "config.sqlite",
        )
        self.assertIsInstance(cfg.news, NewsConfig)
        self.assertTrue(cfg.news.enabled)
        self.assertEqual(cfg.news.scan_interval_minutes, 60)
        self.assertEqual(cfg.news.ttl_hours, 24)
        self.assertIn("stocktwits", cfg.news.enabled_sources)

    def test_toml_overrides_defaults(self) -> None:
        self._write_config(
            """
            [mode]
            trading = "paper"

            [news]
            enabled = false
            scan_interval_minutes = 15
            ttl_hours = 48
            enabled_sources = ["stocktwits", "yfinance"]
            google_news_max_items_per_query = 5
            """
        )
        cfg = load_config(
            project_root=self.tmp,
            env_path=self.tmp / ".env",
            config_path=self.tmp / "config.toml",
            config_db_path=self.tmp / "config.sqlite",
        )
        self.assertFalse(cfg.news.enabled)
        self.assertEqual(cfg.news.scan_interval_minutes, 15)
        self.assertEqual(cfg.news.ttl_hours, 48)
        self.assertEqual(cfg.news.enabled_sources, ("stocktwits", "yfinance"))
        self.assertEqual(cfg.news.google_news_max_items_per_query, 5)

    def test_db_override_wins_over_toml(self) -> None:
        toml_path = self._write_config(
            """
            [mode]
            trading = "paper"

            [news]
            scan_interval_minutes = 30
            """
        )
        db_path = self.tmp / "config.sqlite"
        store = open_store(db_path)
        try:
            store.set_section("news", {"scan_interval_minutes": 90})
        finally:
            store.close()

        cfg = load_config(
            project_root=self.tmp,
            env_path=self.tmp / ".env",
            config_path=toml_path,
            config_db_path=db_path,
        )
        # DB value wins.
        self.assertEqual(cfg.news.scan_interval_minutes, 90)


if __name__ == "__main__":
    unittest.main()
