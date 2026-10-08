"""Word glosses from the JMdict subset built by scripts/build_index.py.

Two rules keep the glosses sensible:
1. Particles and auxiliaries (は, を, です, ます...) use a fixed table. A plain
   dictionary lookup of は finds 羽 "feather", which is wrong in context.
2. When a spelling has several entries (彼 = かれ "he" / あれ "that"), pick
   the one whose reading matches SudachiPy's reading of the token.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .nlp import kata_to_hira

log = logging.getLogger(__name__)

# Keyed by dictionary form (base). Grammar words get role descriptions, not translations.
FUNCTION_WORDS = {
    "は": "topic marker",
    "が": "subject marker",
    "を": "object marker",
    "に": "to / at / in",
    "へ": "toward",
    "で": "at / by means of",
    "と": "and / with",
    "も": "also",
    "の": "'s (possessive)",
    "から": "from / because",
    "まで": "until",
    "より": "than",
    "や": "and (among others)",
    "か": "question marker",
    "ね": "right? / isn't it",
    "よ": "(emphasis)",
    "な": "(emphasis / don't)",
    "て": "te-form connector",
    "ば": "if",
    "けど": "but",
    "けれど": "but",
    "のに": "even though",
    "ながら": "while",
    "ので": "because",
    "ます": "polite ending",
    "です": "is (polite)",
    "だ": "is",
    "た": "past tense",
    "ない": "not",
    "ぬ": "not",
    "ん": "not",
    "たい": "want to",
    "れる": "passive / potential",
    "られる": "passive / potential",
    "せる": "make / let (causative)",
    "させる": "make / let (causative)",
    "う": "let's / volitional",
    "よう": "let's / volitional",
    "そう": "looks like / I hear",
    "らしい": "seems",
    "ちゃう": "end up doing",
}
FUNCTION_POS = {"particle", "auxiliary"}


class Dictionary:
    def __init__(self, words: dict | None = None):
        self.words = words or {}  # word -> [{"r": [readings], "g": [glosses]}, ...]

    @property
    def entries(self) -> dict:  # kept for /health
        return self.words

    @classmethod
    def from_dir(cls, data_dir: Path) -> "Dictionary":
        path = Path(data_dir) / "dictionary.json"
        if not path.exists():
            log.info("No dictionary.json found; word glosses disabled")
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 2:
            log.warning("dictionary.json is in the old format; rerun scripts/build_index.py")
            # old format: word -> {"readings": [...], "glosses": [...]}
            return cls({w: [{"r": e["readings"], "g": e["glosses"]}] for w, e in data.items()})
        return cls(data["words"])

    def lookup(self, base: str, surface: str = "", reading: str = "", pos: str = "") -> list[str]:
        if pos in FUNCTION_POS:
            gloss = FUNCTION_WORDS.get(base) or FUNCTION_WORDS.get(surface)
            return [gloss] if gloss else []
        reading = kata_to_hira(reading)
        for key in (base, surface):
            candidates = self.words.get(key) if key else None
            if not candidates:
                continue
            for entry in candidates:
                if reading and reading in {kata_to_hira(r) for r in entry["r"]}:
                    return entry["g"]
            # Reading didn't match exactly (e.g. reading of a conjugated surface): first entry
            return candidates[0]["g"]
        return []
