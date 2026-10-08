"""Tiny SQLite response cache.

Learners ask about the same sentences and grammar points again and again, so
caching LLM answers saves a lot of free-tier quota and makes repeat requests
instant.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path | str, ttl_seconds: int = 30 * 24 * 3600):
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)"
        )
        self._db.commit()

    @staticmethod
    def make_key(kind: str, payload: dict) -> str:
        raw = kind + "|" + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT value, created FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        value, created = row
        if time.time() - created > self.ttl:
            return None
        return json.loads(value)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )
            self._db.commit()
