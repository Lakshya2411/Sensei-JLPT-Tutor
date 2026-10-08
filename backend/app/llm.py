"""LLM access with a free-tier-friendly fallback chain.

Both Gemini and Groq expose OpenAI-compatible endpoints, so one small client
covers both. We try Gemini first; if it fails (rate limit, outage, missing
key) we fall back to Groq. Uses only the standard library, so there's no SDK
to install and the HTTP layer is easy to mock in tests.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .config import settings

log = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_BASE = "https://api.groq.com/openai/v1"


class LLMError(RuntimeError):
    pass


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str


def default_providers() -> list[Provider]:
    providers = []
    if settings.gemini_api_key:
        providers.append(Provider("gemini", GEMINI_BASE, settings.gemini_api_key, settings.gemini_model))
    if settings.groq_api_key:
        providers.append(Provider("groq", GROQ_BASE, settings.groq_api_key, settings.groq_model))
    return providers


# (url, headers, payload, timeout) -> parsed JSON response. Swappable in tests.
HttpPost = Callable[[str, dict, dict, float], dict]


def _http_post(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise LLMError(f"HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"network error: {e}") from e


def parse_json(text: str) -> dict:
    """Pull a JSON object out of a model reply, tolerating ```json fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class LLMClient:
    def __init__(
        self,
        providers: list[Provider] | None = None,
        http_post: HttpPost = _http_post,
        timeout: float = 30.0,
        retries_per_provider: int = 1,
    ):
        self.providers = default_providers() if providers is None else providers
        self.http_post = http_post
        self.timeout = timeout
        self.retries = retries_per_provider

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def chat_json(self, system: str, user: str, temperature: float = 0.3) -> tuple[dict, str]:
        """Ask for a JSON object. Returns (parsed_json, provider_name)."""
        if not self.providers:
            raise LLMError("No LLM API key configured. Set GEMINI_API_KEY and/or GROQ_API_KEY.")
        errors = []
        for p in self.providers:
            payload = {
                "model": p.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            headers = {"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"}
            for attempt in range(self.retries + 1):
                try:
                    data = self.http_post(f"{p.base_url}/chat/completions", headers, payload, self.timeout)
                    content = data["choices"][0]["message"]["content"]
                    return parse_json(content), p.name
                except (LLMError, KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
                    errors.append(f"{p.name}: {e}")
                    log.warning("LLM call failed (%s, attempt %d): %s", p.name, attempt + 1, e)
                    is_rate_limit = "HTTP 429" in str(e)
                    if attempt < self.retries and not is_rate_limit:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        break  # rate limited or out of retries: move to next provider
        raise LLMError("All LLM providers failed: " + " | ".join(errors))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings via Gemini's OpenAI-compatible endpoint."""
        gemini = next((p for p in self.providers if p.name == "gemini"), None)
        if gemini is None:
            raise LLMError("Embeddings need GEMINI_API_KEY.")
        headers = {"Authorization": f"Bearer {gemini.api_key}", "Content-Type": "application/json"}
        payload = {"model": settings.gemini_embed_model, "input": texts}
        data = self.http_post(f"{gemini.base_url}/embeddings", headers, payload, self.timeout)
        return [row["embedding"] for row in sorted(data["data"], key=lambda r: r.get("index", 0))]
