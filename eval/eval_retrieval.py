"""Evaluate grammar retrieval: recall@k and MRR, per retrieval method.

Usage (from the project root):
    python eval/eval_retrieval.py
    python eval/eval_retrieval.py --k 3

Compares: BM25 only, markers only, hybrid (markers + BM25), and hybrid +
embeddings when grammar_vectors.npy exists and GEMINI_API_KEY is set.
This table is what goes in the README and on your resume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.retriever import GrammarRetriever  # noqa: E402


def evaluate(rank_fn, questions, k):
    hits, rr = 0, 0.0
    for q in questions:
        ranking = rank_fn(q["query"])
        expected = set(q["expected"])
        if expected & set(ranking[:k]):
            hits += 1
        for i, gid in enumerate(ranking):
            if gid in expected:
                rr += 1 / (i + 1)
                break
    n = len(questions)
    return hits / n, rr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    questions = json.loads((ROOT / "eval" / "retrieval_questions.json").read_text(encoding="utf-8"))
    embedder = None
    try:
        from app.llm import LLMClient

        client = LLMClient()
        if any(p.name == "gemini" for p in client.providers):
            embedder = client
    except Exception:
        pass
    r = GrammarRetriever.from_files(ROOT / "data/grammar/n5_n4_grammar.json", ROOT / "data/processed", embedder)

    methods = {
        "BM25 only": r.bm25_ranking,
        "Markers only": r.marker_ranking,
        "Hybrid (markers + BM25)": lambda q: [h.grammar["id"] for h in r.search(q, k=10, use_embeddings=False)],
    }
    if r.grammar_vectors is not None and embedder is not None:
        methods["Embeddings only"] = r.embedding_ranking
        methods["Hybrid + embeddings"] = lambda q: [h.grammar["id"] for h in r.search(q, k=10)]

    print(f"{len(questions)} questions, {len(r.grammar)} grammar points, {len(r.examples)} mined examples\n")
    print(f"{'Method':<28}{'Recall@' + str(args.k):>10}{'MRR':>8}")
    print("-" * 46)
    for name, fn in methods.items():
        recall, mrr = evaluate(fn, questions, args.k)
        print(f"{name:<28}{recall:>10.2f}{mrr:>8.2f}")

    if args.show_misses:
        fn = list(methods.values())[-1]
        print("\nMisses for", list(methods)[-1])
        for q in questions:
            top = fn(q["query"])[: args.k]
            if not set(q["expected"]) & set(top):
                print(f'  {q["query"]}  expected={q["expected"]}  got={top}')


if __name__ == "__main__":
    main()
