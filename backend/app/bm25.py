"""A small BM25 (Okapi) implementation.

Written by hand instead of using a library so the scoring is easy to explain:

    score(q, d) = Σ  idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
                 t∈q

k1 controls how quickly repeated terms stop adding score, b controls how much
long documents are penalised.
"""
from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tf = [Counter(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in corpus]
        self.n_docs = len(corpus)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        df = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        # BM25+ style idf that never goes negative
        self.idf = {t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n_docs
        q_terms = set(query)
        for i, tf in enumerate(self.doc_tf):
            if not tf:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            s = 0.0
            for t in q_terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out
