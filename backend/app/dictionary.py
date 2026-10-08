"""Word lookups from the JMdict subset built by scripts/build_index.py."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Dictionary:
    def __init__(self, entries: dict | None = None):
        self.entries = entries or {}

    @classmethod
    def from_dir(cls, data_dir: Path) -> "Dictionary":
        path = Path(data_dir) / "dictionary.json"
        if not path.exists():
            log.info("No dictionary.json found; word glosses disabled")
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def lookup(self, base: str, surface: str = "", reading: str = "") -> list[str]:
        for key in (base, surface, reading):
            if key and key in self.entries:
                return self.entries[key]["glosses"]
        return []
