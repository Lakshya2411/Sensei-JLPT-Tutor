"""Build the processed knowledge base from the raw data.

Usage (from the project root):
    python scripts/build_index.py              # examples + dictionary
    python scripts/build_index.py --embed      # also precompute Gemini embeddings

Outputs (data/processed/):
- examples.jsonl     Tatoeba sentences attached to the grammar point they use
- dictionary.json    common JMdict words -> readings + English glosses
- grammar_vectors.npy  (with --embed) one embedding per grammar point
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
GRAMMAR = ROOT / "data" / "grammar" / "n5_n4_grammar.json"

sys.path.insert(0, str(ROOT / "backend"))


# ---------------- Tatoeba examples ----------------
def read_tatoeba(path: Path):
    """ManyThings format: English<TAB>Japanese<TAB>attribution."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield {
                    "en": parts[0].strip(),
                    "ja": unicodedata.normalize("NFKC", parts[1]).strip(),
                    "attribution": parts[2].strip() if len(parts) > 2 else "Tatoeba (CC BY 2.0 FR)",
                }


def mine_examples(grammar: list[dict], pairs, per_point: int = 40, max_len: int = 30) -> list[dict]:
    """Attach short sentences to every grammar point whose markers they match.

    Preference goes to short sentences and to sentences that match fewer
    grammar points, since those are clearer examples for learners.
    """
    markers = {g["id"]: [re.compile(m) for m in g.get("markers", [])] for g in grammar}
    candidates: dict[str, list[tuple]] = {g["id"]: [] for g in grammar}
    seen = set()
    for p in pairs:
        ja = p["ja"]
        if len(ja) > max_len or ja in seen:
            continue
        seen.add(ja)
        hits = [gid for gid, pats in markers.items() if any(m.search(ja) for m in pats)]
        for gid in hits:
            candidates[gid].append((len(hits), len(ja), p))
    out = []
    for gid, cands in candidates.items():
        cands.sort(key=lambda c: (c[0], c[1]))
        for _, _, p in cands[:per_point]:
            out.append({"grammar_id": gid, **p})
    return out


# ---------------- JMdict dictionary ----------------
COMMON_TAGS = {"news1", "ichi1", "spec1", "spec2", "gai1"}


def build_dictionary(xml_path: Path) -> dict:
    """Keep only common words (priority tags) so the file stays small."""
    entries: dict[str, dict] = {}
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != "entry":
            continue
        kanji = [(k.findtext("keb"), {p.text for p in k.findall("ke_pri")}) for k in el.findall("k_ele")]
        readings = [(r.findtext("reb"), {p.text for p in r.findall("re_pri")}) for r in el.findall("r_ele")]
        is_common = any(tags & COMMON_TAGS for _, tags in kanji + readings)
        if is_common:
            glosses = []
            for sense in el.findall("sense"):
                glosses += [g.text for g in sense.findall("gloss") if g.text]
                if len(glosses) >= 4:
                    break
            info = {"readings": [r for r, _ in readings][:3], "glosses": glosses[:4]}
            for word, _ in kanji + readings:
                if word and word not in entries:
                    entries[word] = info
        el.clear()
    return entries


# ---------------- embeddings ----------------
def embed_grammar(grammar: list[dict]) -> None:
    import numpy as np
    from app.llm import LLMClient
    from app.retriever import grammar_doc_text

    client = LLMClient()
    texts = [grammar_doc_text(g) for g in grammar]
    vectors = []
    for i in range(0, len(texts), 50):
        vectors += client.embed(texts[i : i + 50])
    np.save(OUT / "grammar_vectors.npy", np.asarray(vectors, dtype="float32"))
    print(f"Saved {len(vectors)} grammar embeddings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="precompute Gemini embeddings (needs GEMINI_API_KEY)")
    ap.add_argument("--per-point", type=int, default=40, help="max example sentences per grammar point")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    grammar = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    print(f"{len(grammar)} grammar points")

    tatoeba = RAW / "jpn.txt"
    if tatoeba.exists():
        examples = mine_examples(grammar, read_tatoeba(tatoeba), args.per_point)
        with (OUT / "examples.jsonl").open("w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} example sentences -> data/processed/examples.jsonl")
    else:
        print("No data/raw/jpn.txt (run scripts/download_data.py tatoeba); skipping examples")

    jmdict = RAW / "JMdict_e.xml"
    if jmdict.exists():
        d = build_dictionary(jmdict)
        (OUT / "dictionary.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(d)} dictionary headwords -> data/processed/dictionary.json")
    else:
        print("No data/raw/JMdict_e.xml (run scripts/download_data.py jmdict); skipping dictionary")

    if args.embed:
        embed_grammar(grammar)


if __name__ == "__main__":
    main()
