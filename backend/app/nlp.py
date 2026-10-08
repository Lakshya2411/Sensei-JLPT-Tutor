"""Japanese text analysis.

Japanese is written without spaces, so we use SudachiPy (a morphological
analyzer) to split text into words and get each word's reading, dictionary
form and part of speech.

If SudachiPy isn't installed (e.g. a very small server), we fall back to
character bigrams. Bigrams are a common trick for search over CJK text: they
are good enough for BM25 retrieval, but they can't give readings or POS.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from functools import lru_cache

_JP_CHAR = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ々〆ー]")
_LATIN_WORD = re.compile(r"[a-z0-9]+")
_PUNCT_POS = {"補助記号", "空白"}


@dataclass
class Token:
    surface: str
    reading: str  # hiragana
    base: str  # dictionary form, e.g. 食べ → 食べる
    pos: str  # coarse part of speech in English
    pos_ja: str

    def to_dict(self) -> dict:
        return asdict(self)


_POS_EN = {
    "名詞": "noun",
    "代名詞": "pronoun",
    "動詞": "verb",
    "形容詞": "i-adjective",
    "形状詞": "na-adjective",
    "副詞": "adverb",
    "助詞": "particle",
    "助動詞": "auxiliary",
    "接続詞": "conjunction",
    "連体詞": "pre-noun adjectival",
    "感動詞": "interjection",
    "接頭辞": "prefix",
    "接尾辞": "suffix",
    "補助記号": "symbol",
    "記号": "symbol",
    "空白": "space",
}


def kata_to_hira(text: str) -> str:
    """Convert katakana to hiragana (ァ..ヶ are exactly 0x60 above ぁ..ゖ)."""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def normalize(text: str) -> str:
    """NFKC turns full-width ASCII/half-width katakana into standard forms."""
    return unicodedata.normalize("NFKC", text).strip()


@lru_cache(maxsize=1)
def _sudachi():
    """Load SudachiPy once. Returns None if it isn't installed."""
    try:
        from sudachipy import Dictionary, SplitMode
    except ImportError:
        return None
    for dict_name in ("small", "core", "full"):
        try:
            return Dictionary(dict=dict_name).create(), SplitMode.C
        except Exception:
            continue
    return None


def sudachi_available() -> bool:
    return _sudachi() is not None


def analyze(text: str) -> list[Token]:
    """Split text into tokens with readings. Empty list if SudachiPy is missing."""
    loaded = _sudachi()
    if loaded is None:
        return []
    tokenizer, mode = loaded
    tokens = []
    for m in tokenizer.tokenize(normalize(text), mode):
        pos_ja = m.part_of_speech()[0]
        if pos_ja in _PUNCT_POS:
            continue
        surface = m.surface()
        reading = kata_to_hira(m.reading_form())
        tokens.append(
            Token(
                surface=surface,
                reading=reading,
                base=m.dictionary_form(),
                pos=_POS_EN.get(pos_ja, pos_ja),
                pos_ja=pos_ja,
            )
        )
    return tokens


def _bigrams(text: str) -> list[str]:
    out = []
    for run in re.findall(r"[぀-ヿ㐀-鿿ｦ-ﾟ々〆ー]+", text):
        if len(run) == 1:
            out.append(run)
        out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def search_terms(text: str) -> list[str]:
    """Terms used for BM25 indexing and querying.

    With SudachiPy: surface forms + dictionary forms (so 食べた matches 食べる),
    plus bigrams so grammar patterns like ている still match across word
    boundaries. Without SudachiPy: bigrams only. English words are lowercased
    and kept as-is in both cases.
    """
    text = normalize(text).lower()
    terms: list[str] = []
    loaded = _sudachi()
    if loaded is not None:
        tokenizer, mode = loaded
        for m in tokenizer.tokenize(text, mode):
            if m.part_of_speech()[0] in _PUNCT_POS or not has_japanese(m.surface()):
                continue  # English words are added below
            terms.append(m.surface())
            if m.dictionary_form() != m.surface():
                terms.append(m.dictionary_form())
    terms.extend(_bigrams(text))
    terms.extend(_LATIN_WORD.findall(text))
    return terms


def has_japanese(text: str) -> bool:
    return bool(_JP_CHAR.search(text))
