"""Hybrid retrieval over the grammar knowledge base.

Three signals, fused with Reciprocal Rank Fusion (RRF):

1. Pattern markers - each grammar entry lists regexes for its surface forms
   (e.g. 〜ている matches ている/ていた/ています). Exact, high precision.
2. BM25 - keyword search over grammar notes AND the example sentences
   attached to each grammar point (from Tatoeba), so a query that resembles
   an example sentence pulls in its grammar point.
3. Embeddings (optional) - semantic search, so English questions like
   "how do I say I've done something before" find 〜たことがある.

RRF: score(d) = Σ_lists weight / (k + rank(d)). It needs no score
normalisation, which is why it's a common way to merge BM25 and vector search.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25
from .nlp import search_terms

log = logging.getLogger(__name__)

LEVEL_ORDER = {"N5": 5, "N4": 4, "N3": 3, "N2": 2, "N1": 1}
RRF_K = 60


@dataclass
class Hit:
    grammar: dict
    score: float
    matched_by: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        g = self.grammar
        return {
            "id": g["id"],
            "pattern": g["pattern"],
            "level": g["level"],
            "meaning": g["meaning"],
            "formation": g.get("formation", ""),
            "explanation": g.get("explanation", ""),
            "examples": (g.get("examples", []) + self.examples)[:4],
            "score": round(self.score, 4),
            "matched_by": self.matched_by,
        }


def grammar_doc_text(g: dict) -> str:
    parts = [g["pattern"], g["meaning"], g.get("formation", ""), g.get("explanation", "")]
    parts += g.get("aliases", [])
    for ex in g.get("examples", []):
        parts += [ex.get("ja", ""), ex.get("en", "")]
    return " ".join(p for p in parts if p)


class GrammarRetriever:
    def __init__(self, grammar: list[dict], examples: list[dict] | None = None, embedder=None):
        """
        grammar:  entries from data/grammar/*.json
        examples: optional [{"grammar_id", "ja", "en"}] mined from Tatoeba
        embedder: optional object with .embed(list[str]) -> list[vectors]
        """
        self.grammar = grammar
        self.by_id = {g["id"]: g for g in grammar}
        self.examples = [e for e in (examples or []) if e.get("grammar_id") in self.by_id]
        self.examples_by_gid: dict[str, list[dict]] = {}
        for e in self.examples:
            self.examples_by_gid.setdefault(e["grammar_id"], []).append(e)

        self.markers = {
            g["id"]: [re.compile(m) for m in g.get("markers", [])] for g in grammar
        }

        # One BM25 index over grammar docs + example docs; doc_gid maps doc -> grammar id
        texts = [grammar_doc_text(g) for g in grammar] + [
            f'{e["ja"]} {e.get("en", "")}' for e in self.examples
        ]
        self.doc_gid = [g["id"] for g in grammar] + [e["grammar_id"] for e in self.examples]
        self.bm25 = BM25([search_terms(t) for t in texts])

        self.embedder = embedder
        self.grammar_vectors = None  # numpy array [n_grammar, dim], set via load_vectors

    # ---------- construction helpers ----------
    @classmethod
    def from_files(cls, grammar_file: Path, data_dir: Path, embedder=None) -> "GrammarRetriever":
        grammar = json.loads(Path(grammar_file).read_text(encoding="utf-8"))
        examples = []
        ex_path = Path(data_dir) / "examples.jsonl"
        if ex_path.exists():
            with ex_path.open(encoding="utf-8") as f:
                examples = [json.loads(line) for line in f if line.strip()]
        r = cls(grammar, examples, embedder)
        r.load_vectors(Path(data_dir) / "grammar_vectors.npy")
        log.info("Retriever ready: %d grammar points, %d examples, vectors=%s",
                 len(grammar), len(r.examples), r.grammar_vectors is not None)
        return r

    def load_vectors(self, path: Path) -> None:
        if not path.exists():
            return
        import numpy as np

        vecs = np.load(path)
        if vecs.shape[0] != len(self.grammar):
            log.warning("Ignoring %s: %d vectors for %d grammar points (rebuild the index)",
                        path, vecs.shape[0], len(self.grammar))
            return
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.grammar_vectors = vecs / np.where(norms == 0, 1, norms)

    # ---------- the three signals ----------
    def marker_ranking(self, query: str) -> list[str]:
        """Grammar ids whose markers occur in the query, longest match first."""
        found = []
        for gid, patterns in self.markers.items():
            best = max((len(m.group(0)) for p in patterns for m in p.finditer(query)), default=0)
            if best:
                found.append((best, gid))
        return [gid for _, gid in sorted(found, key=lambda x: -x[0])]

    def bm25_ranking(self, query: str) -> list[str]:
        scores = self.bm25.scores(search_terms(query))
        best: dict[str, float] = {}
        for gid, s in zip(self.doc_gid, scores):
            if s > 0 and s > best.get(gid, 0):
                best[gid] = s
        return sorted(best, key=lambda gid: -best[gid])

    def embedding_ranking(self, query: str) -> list[str]:
        if self.embedder is None or self.grammar_vectors is None:
            return []
        import numpy as np

        try:
            q = np.asarray(self.embedder.embed([query])[0], dtype=float)
        except Exception as e:  # free tier hiccup: just skip this signal
            log.warning("Query embedding failed, using BM25 + markers only: %s", e)
            return []
        q = q / (np.linalg.norm(q) or 1)
        sims = self.grammar_vectors @ q
        order = np.argsort(-sims)
        return [self.grammar[i]["id"] for i in order[:20]]

    # ---------- fusion ----------
    def search(self, query: str, k: int = 5, max_level: str | None = None,
               use_embeddings: bool = True) -> list[Hit]:
        rankings = {
            "pattern": (self.marker_ranking(query), 2.0),
            "bm25": (self.bm25_ranking(query), 1.0),
        }
        if use_embeddings:
            rankings["embedding"] = (self.embedding_ranking(query), 1.0)

        allowed = None
        if max_level:
            floor = LEVEL_ORDER.get(max_level.upper(), 1)
            allowed = {g["id"] for g in self.grammar if LEVEL_ORDER.get(g["level"], 1) >= floor}

        fused: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        for name, (ranking, weight) in rankings.items():
            for rank, gid in enumerate(ranking):
                if allowed is not None and gid not in allowed:
                    continue
                fused[gid] = fused.get(gid, 0.0) + weight / (RRF_K + rank + 1)
                sources.setdefault(gid, []).append(name)

        top = sorted(fused, key=lambda gid: -fused[gid])[:k]
        return [
            Hit(self.by_id[gid], fused[gid], sources[gid], self.examples_by_gid.get(gid, [])[:3])
            for gid in top
        ]

    def detect(self, sentence: str, max_level: str | None = None) -> list[Hit]:
        """Grammar points that actually appear in a Japanese sentence (markers only)."""
        ids = self.marker_ranking(sentence)
        if max_level:
            floor = LEVEL_ORDER.get(max_level.upper(), 1)
            ids = [i for i in ids if LEVEL_ORDER.get(self.by_id[i]["level"], 1) >= floor]
        return [Hit(self.by_id[i], 1.0, ["pattern"], self.examples_by_gid.get(i, [])[:3]) for i in ids]
