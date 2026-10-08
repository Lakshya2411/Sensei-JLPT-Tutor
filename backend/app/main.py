"""Sensei API.

Run locally:   uvicorn app.main:app --reload --port 8000   (from the backend/ folder)
Docs:          http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import nlp, prompts
from .cache import ResponseCache
from .config import settings
from .dictionary import Dictionary
from .llm import LLMClient, LLMError
from .ratelimit import RateLimiter
from .retriever import GrammarRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Sensei - JLPT tutor API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------- services (created once, lazily) ----------
@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()


@lru_cache(maxsize=1)
def get_retriever() -> GrammarRetriever:
    llm = get_llm()
    embedder = llm if any(p.name == "gemini" for p in llm.providers) else None
    return GrammarRetriever.from_files(settings.grammar_file, settings.data_dir, embedder)


@lru_cache(maxsize=1)
def get_dictionary() -> Dictionary:
    return Dictionary.from_dir(settings.data_dir)


def analyze_tokens(text: str) -> list[dict]:
    """SudachiPy tokens plus JMdict glosses (no LLM involved)."""
    d = get_dictionary()
    out = []
    for t in nlp.analyze(text):
        row = t.to_dict()
        row["gloss"] = d.lookup(t.base, t.surface, t.reading, t.pos)[:2]
        out.append(row)
    return out


@lru_cache(maxsize=1)
def get_cache() -> ResponseCache:
    return ResponseCache(settings.cache_path)


limiter = RateLimiter(settings.rate_limit_per_min)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # set by HF Spaces / Render proxies
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    if not limiter.allow(client_ip(request)):
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")


def ask_llm(kind: str, cache_payload: dict, system: str, user: str, temperature: float = 0.3) -> dict:
    cache = get_cache()
    key = ResponseCache.make_key(kind, cache_payload)
    cached = cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}
    try:
        result, provider = get_llm().chat_json(system, user, temperature)
    except LLMError as e:
        raise HTTPException(503, f"The AI tutor is unavailable right now: {e}") from e
    result["provider"] = provider
    cache.set(key, result)
    return {**result, "cached": False}


# ---------- request models ----------
class SentenceIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=300)
    max_level: str | None = Field(None, pattern="^N[1-5]$")


class SearchIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=300)
    k: int = Field(5, ge=1, le=10)
    max_level: str | None = Field(None, pattern="^N[1-5]$")


class PracticeIn(BaseModel):
    grammar_id: str
    n: int = Field(5, ge=1, le=10)


# ---------- routes ----------
@app.get("/health")
def health():
    r = get_retriever()
    return {
        "status": "ok",
        "sudachi": nlp.sudachi_available(),
        "llm_providers": [p.name for p in get_llm().providers],
        "grammar_points": len(r.grammar),
        "example_sentences": len(r.examples),
        "embeddings": r.grammar_vectors is not None,
        "dictionary_words": len(get_dictionary().entries),
    }


@app.get("/grammar")
def list_grammar(level: str | None = None):
    items = get_retriever().grammar
    if level:
        items = [g for g in items if g["level"] == level.upper()]
    return [{"id": g["id"], "pattern": g["pattern"], "level": g["level"], "meaning": g["meaning"]} for g in items]


@app.get("/grammar/{grammar_id}")
def get_grammar(grammar_id: str):
    r = get_retriever()
    g = r.by_id.get(grammar_id)
    if not g:
        raise HTTPException(404, "Unknown grammar id")
    return {**g, "mined_examples": r.examples_by_gid.get(grammar_id, [])[:10]}


@app.post("/analyze")
def analyze(body: SentenceIn):
    """No LLM: tokens with readings + grammar patterns detected in the sentence."""
    tokens = analyze_tokens(body.text)
    detected = get_retriever().detect(nlp.normalize(body.text), body.max_level)
    return {"text": body.text, "tokens": tokens, "grammar": [h.to_dict() for h in detected]}


@app.post("/search")
def search(body: SearchIn):
    """Retrieval only (useful for debugging and evaluation)."""
    hits = get_retriever().search(body.query, k=body.k, max_level=body.max_level)
    return {"query": body.query, "results": [h.to_dict() for h in hits]}


@app.post("/explain")
def explain(body: SentenceIn, request: Request):
    check_rate_limit(request)
    text = nlp.normalize(body.text)
    if not nlp.has_japanese(text):
        raise HTTPException(400, "Please enter a Japanese sentence.")
    tokens = analyze_tokens(text)
    r = get_retriever()
    detected = r.detect(text, body.max_level)
    extra = [h for h in r.search(text, k=4, max_level=body.max_level) if h.grammar["id"] not in {d.grammar["id"] for d in detected}]
    hits = [h.to_dict() for h in (detected + extra)[:6]]

    system, user = prompts.explain_prompt(text, tokens, hits)
    result = ask_llm("explain", {"text": text}, system, user)
    result["grammar"] = prompts.keep_known_ids(result.get("grammar", []), {h["id"] for h in hits})
    return {"text": text, "tokens": tokens, "sources": hits, "answer": result}


@app.post("/practice")
def practice(body: PracticeIn, request: Request):
    check_rate_limit(request)
    r = get_retriever()
    g = r.by_id.get(body.grammar_id)
    if not g:
        raise HTTPException(404, "Unknown grammar id")
    examples = (g.get("examples", []) + r.examples_by_gid.get(g["id"], []))[:5]
    system, user = prompts.practice_prompt(g, body.n, examples)
    result = ask_llm("practice", {"id": g["id"], "n": body.n}, system, user, temperature=0.7)
    return {"grammar": {"id": g["id"], "pattern": g["pattern"], "level": g["level"]}, "answer": result}


@app.post("/correct")
def correct(body: SentenceIn, request: Request):
    check_rate_limit(request)
    text = nlp.normalize(body.text)
    if not nlp.has_japanese(text):
        raise HTTPException(400, "Please enter a Japanese sentence.")
    tokens = analyze_tokens(text)
    hits = [h.to_dict() for h in get_retriever().search(text, k=5)]
    system, user = prompts.correct_prompt(text, tokens, hits)
    result = ask_llm("correct", {"text": text}, system, user, temperature=0.1)
    result["errors"] = prompts.keep_known_ids(result.get("errors", []), {h["id"] for h in hits}, key="grammar_id")
    return {"text": text, "sources": hits, "answer": result}
