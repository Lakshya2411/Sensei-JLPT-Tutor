"""Prompt templates. Every prompt is grounded in retrieved grammar notes and
asks for strict JSON so the frontend can render it reliably."""
from __future__ import annotations

import json

TUTOR_PERSONA = (
    "You are Sensei, a patient Japanese tutor for JLPT N5-N3 learners whose native "
    "language is English. Keep explanations short and concrete. Use only the grammar "
    "notes provided as your reference; if they don't cover something, say so briefly "
    "instead of guessing. Always reply with a single JSON object and nothing else."
)


def format_notes(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        ex = "; ".join(f'{e["ja"]} ({e.get("en", "")})' for e in h.get("examples", [])[:2])
        blocks.append(
            f'[{h["id"]}] {h["pattern"]} ({h["level"]}): {h["meaning"]}\n'
            f'  Formation: {h.get("formation", "")}\n'
            f'  Notes: {h.get("explanation", "")}\n'
            f"  Examples: {ex}"
        )
    return "\n".join(blocks) if blocks else "(no matching grammar notes)"


def explain_prompt(sentence: str, tokens: list[dict], hits: list[dict]) -> tuple[str, str]:
    token_str = " / ".join(
        f'{t["surface"]}[{t["reading"]}, {t["pos"]}' + (f', "{t["gloss"][0]}"' if t.get("gloss") else "") + "]"
        for t in tokens
    ) or "(not available)"
    user = f"""Explain this Japanese sentence to a learner.

Sentence: {sentence}
Morphological analysis (from SudachiPy): {token_str}

Grammar notes (cite them by id):
{format_notes(hits)}

Return JSON with exactly these keys:
{{
  "translation": "natural English translation",
  "literal": "word-for-word gloss",
  "words": [{{"surface": "...", "reading": "hiragana", "meaning": "short English gloss"}}],
  "grammar": [{{"id": "grammar note id from the list above", "pattern": "...", "how_it_is_used_here": "1-2 sentences"}}],
  "tip": "one short learning tip"
}}
Only include grammar ids that appear in the notes above and are actually used in the sentence."""
    return TUTOR_PERSONA, user


def practice_prompt(grammar: dict, n: int, examples: list[dict]) -> tuple[str, str]:
    user = f"""Create {n} practice questions for this grammar point.

Grammar note:
{format_notes([{**grammar, "examples": examples or grammar.get("examples", [])}])}

Mix these types: "fill_blank" (sentence with ＿＿ where the pattern goes), "choose" (pick the correct form),
"translate" (English to Japanese). Use vocabulary a {grammar["level"]} learner would know.

Return JSON:
{{
  "questions": [
    {{
      "type": "fill_blank | choose | translate",
      "question": "the question text",
      "options": ["only for choose and fill_blank: 3-4 options, otherwise empty list"],
      "answer": "the correct answer",
      "explanation": "why, in one or two sentences"
    }}
  ]
}}"""
    return TUTOR_PERSONA, user


def correct_prompt(sentence: str, tokens: list[dict], hits: list[dict]) -> tuple[str, str]:
    token_str = " / ".join(f'{t["surface"]}[{t["pos"]}]' for t in tokens) or "(not available)"
    user = f"""A learner wrote this Japanese sentence. Check it for grammar, particle, conjugation and word-choice mistakes.

Learner's sentence: {sentence}
Morphological analysis: {token_str}

Possibly relevant grammar notes (cite by id when a mistake relates to one):
{format_notes(hits)}

Return JSON:
{{
  "is_correct": true or false,
  "corrected": "the corrected sentence (same as input if correct)",
  "translation": "English meaning of the corrected sentence",
  "errors": [{{"original": "wrong part", "fix": "corrected part", "explanation": "short reason", "grammar_id": "id or null"}}],
  "more_natural": "optional more natural phrasing, or empty string"
}}
Do not invent errors: if the sentence is correct and natural, return an empty errors list."""
    return TUTOR_PERSONA, user


def keep_known_ids(items: list[dict], known: set[str], key: str = "id") -> list[dict]:
    """Drop citations to grammar ids that weren't in the retrieved context (hallucination guard)."""
    out = []
    for it in items or []:
        gid = it.get(key)
        if gid is None or gid in known:
            out.append(it)
        else:
            it = {**it, key: None}
            out.append(it)
    return out


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
