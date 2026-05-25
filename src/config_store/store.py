"""SQLite-backed key/value store for runtime config overrides.

The store is deliberately tiny — schema is two tables (``config`` and
``meta``), values are JSON-encoded scalars/arrays so we round-trip
ints, floats, bools and lists/tuples without per-field schema entries.
All operations are wrapped in a per-instance lock; SQLite itself is
serialized via ``check_same_thread=False`` plus immediate transactions.

Threading model
---------------
The trading loop and the HTTP control server share one process. The
store is opened once at boot and shared by both threads. Writes are
small and infrequent (only when the operator edits a guardrail), so a
single mutex around every call is plenty.

Failure model
-------------
- A missing/corrupt DB file is logged once and the store falls back to
  an in-memory database. Callers always get a usable handle.
- A failed write is logged and swallowed: it must never crash the
  trading loop. The in-memory ``AppConfig`` is still up to date; the
  next successful write or restart will re-sync.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable


_SCHEMA_VERSION = 1

PERSISTED_SECTIONS: tuple[str, ...] = (
    "guardrails",
    "operations",
    "universe",
    "news",
    "fundamentals",
    "strategy",
    "ai",
    "tools",
    "logging",
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS config (
        section    TEXT NOT NULL,
        key        TEXT NOT NULL,
        value      TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (section, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        name  TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)


# ---------------------------------------------------------------------------
# JSON helpers — preserve ints/floats/bools/lists/tuples
# ---------------------------------------------------------------------------

def _encode(value: Any) -> str:
    """Encode a config value for storage. Tuples → JSON arrays."""
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _decode(raw: str) -> Any:
    """Decode a stored value. Returns ``None`` on malformed JSON."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ConfigStore:
    """Tiny SQLite-backed override store.

    Instances are safe to share between threads. Open one at startup,
    pass it to anything that needs to read or persist runtime config,
    and call :meth:`close` at shutdown.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._path = Path(path) if path else Path(":memory:")
        self._log = logger or logging.getLogger("etrader.config_store")
        self._lock = threading.RLock()
        self._conn = self._open_connection()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        try:
            if str(self._path) != ":memory:":
                self._path.parent.mkdir(parents=True, exist_ok=True)
            return sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                isolation_level=None,  # autocommit; we manage txns explicitly
                timeout=5.0,
            )
        except (sqlite3.Error, OSError) as exc:
            self._log.warning(
                "[config_store] cannot open %s (%s) — falling back to :memory:",
                self._path, exc,
            )
            self._path = Path(":memory:")
            return sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)

    def _ensure_schema(self) -> None:
        with self._lock, self._conn:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(name, value) VALUES(?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Section reads
    # ------------------------------------------------------------------

    def has_any(self) -> bool:
        """Return ``True`` if the store contains at least one config row.

        Used to detect "first ever boot" so we can snapshot the merged
        TOML/default config into the DB.
        """
        with self._lock:
            try:
                cur = self._conn.execute("SELECT 1 FROM config LIMIT 1")
                return cur.fetchone() is not None
            except sqlite3.Error as exc:
                self._log.warning("[config_store] has_any failed: %s", exc)
                return False

    def get_section(self, section: str) -> dict[str, Any]:
        """Return ``{field: value}`` for a section. Missing → ``{}``."""
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT key, value FROM config WHERE section = ?",
                    (section,),
                )
                rows = cur.fetchall()
            except sqlite3.Error as exc:
                self._log.warning("[config_store] read %s failed: %s", section, exc)
                return {}
        out: dict[str, Any] = {}
        for key, raw in rows:
            decoded = _decode(raw)
            if decoded is not None or raw == "null":
                out[str(key)] = decoded
        return out

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return every persisted section as ``{section: {field: value}}``."""
        return {section: self.get_section(section) for section in PERSISTED_SECTIONS}

    # ------------------------------------------------------------------
    # Section writes
    # ------------------------------------------------------------------

    def set_field(self, section: str, key: str, value: Any) -> None:
        """Upsert a single ``(section, key)``. Errors are logged + swallowed."""
        self._upsert([(section, key, value)])

    def set_section(self, section: str, fields: dict[str, Any]) -> None:
        """Upsert every ``(key, value)`` in ``fields`` for ``section``."""
        if not fields:
            return
        self._upsert([(section, k, v) for k, v in fields.items()])

    def snapshot_if_empty(self, sections: dict[str, dict[str, Any]]) -> bool:
        """Atomic first-run dump.

        If the store is empty, write every ``(section, field, value)``
        in one transaction and return ``True``. If the store already has
        rows, do nothing and return ``False``.
        """
        with self._lock:
            if self.has_any():
                return False
            self._upsert(_flatten(sections))
            self._set_meta("first_snapshot_unix", str(time.time()))
            return True

    def add_missing_sections(self, sections: dict[str, dict[str, Any]]) -> list[str]:
        """Backfill new sections introduced after first-run snapshot.

        For each ``(section, fields)`` in ``sections``:
        - If the store has *zero* rows for that section, write every
          field in one transaction.
        - If the store already has at least one row, leave it alone
          (existing operator edits are authoritative).

        Returns the list of section names that were backfilled.
        Used when adding a new config block (e.g. ``[fundamentals]``)
        to an existing deployment so operators don't have to delete
        ``data/config.sqlite`` to pick up new defaults.
        """
        added: list[str] = []
        with self._lock:
            for section, fields in sections.items():
                if self._section_has_any(section):
                    continue
                if not fields:
                    continue
                self._upsert([(section, k, v) for k, v in fields.items()])
                added.append(section)
        if added:
            self._set_meta(
                "last_migration_unix",
                f"{time.time()}:{'+'.join(added)}",
            )
        return added

    def _section_has_any(self, section: str) -> bool:
        try:
            cur = self._conn.execute(
                "SELECT 1 FROM config WHERE section = ? LIMIT 1",
                (section,),
            )
            return cur.fetchone() is not None
        except sqlite3.Error as exc:
            self._log.warning("[config_store] section probe failed (%s): %s", section, exc)
            return False

    def delete_field(self, section: str, key: str) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM config WHERE section = ? AND key = ?",
                    (section, key),
                )
            except sqlite3.Error as exc:
                self._log.warning("[config_store] delete failed: %s", exc)

    def clear_all(self) -> None:
        """Wipe every config row. Used by tests and the eventual reset CLI."""
        with self._lock:
            try:
                self._conn.execute("DELETE FROM config")
            except sqlite3.Error as exc:
                self._log.warning("[config_store] clear failed: %s", exc)

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def get_meta(self, name: str) -> str | None:
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT value FROM meta WHERE name = ?", (name,),
                )
                row = cur.fetchone()
                return row[0] if row else None
            except sqlite3.Error:
                return None

    def _set_meta(self, name: str, value: str) -> None:
        try:
            self._conn.execute(
                "INSERT INTO meta(name, value) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (name, value),
            )
        except sqlite3.Error as exc:
            self._log.warning("[config_store] meta write failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal write path
    # ------------------------------------------------------------------

    def _upsert(self, rows: Iterable[tuple[str, str, Any]]) -> None:
        now = time.time()
        encoded = [(section, key, _encode(value), now) for section, key, value in rows]
        if not encoded:
            return
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.executemany(
                    "INSERT INTO config(section, key, value, updated_at) "
                    "VALUES(?, ?, ?, ?) "
                    "ON CONFLICT(section, key) DO UPDATE SET "
                    "  value=excluded.value, updated_at=excluded.updated_at",
                    encoded,
                )
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                self._log.warning("[config_store] write failed: %s", exc)


def _flatten(sections: dict[str, dict[str, Any]]) -> list[tuple[str, str, Any]]:
    out: list[tuple[str, str, Any]] = []
    for section, fields in sections.items():
        for key, value in (fields or {}).items():
            out.append((section, key, value))
    return out


# ---------------------------------------------------------------------------
# Convenience opener
# ---------------------------------------------------------------------------

def open_store(
    path: Path | str,
    *,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> ConfigStore:
    """Open (or create) the default config store at ``path``."""
    return ConfigStore(path, logger=logger)
