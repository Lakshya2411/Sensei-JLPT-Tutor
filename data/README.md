# Data

| Folder | What | In git? |
|---|---|---|
| `grammar/` | Hand-written grammar notes (N5–N4). Original content for this project. | Yes |
| `raw/` | Downloaded datasets (`scripts/download_data.py`) | No (too big) |
| `processed/` | Built by `scripts/build_index.py`, shipped in the Docker image | Yes (small) |

## Sources and licenses

- **Tatoeba** Japanese–English sentence pairs, via the ManyThings.org export.
  License: CC BY 2.0 FR. Each mined example keeps its attribution string.
  https://tatoeba.org · https://www.manythings.org/anki/
- **JMdict** (English edition) from the Electronic Dictionary Research and Development Group.
  License: CC BY-SA 4.0. Only common words (priority tags) are kept.
  https://www.edrdg.org/jmdict/j_jmdict.html

Check each license page yourself before publishing; terms can change.

## Grammar note format

```json
{
  "id": "nagara",                       // stable id, used for citations
  "pattern": "〜ながら",
  "level": "N4",                        // see note on levels below
  "meaning": "...",
  "formation": "...",
  "explanation": "...",
  "aliases": ["while", "nagara"],       // English/romaji terms so English questions match
  "markers": ["ながら"],                 // regexes that detect the pattern in a sentence
  "examples": [{"ja": "...", "en": "..."}]
}
```

**Levels:** the JLPT has not published official grammar lists since 2010, so levels follow
common textbook conventions and are approximate.

**Markers:** keep them precise. `backend/tests/test_core.py` checks that every example
matches its own markers and that known false positives (like たとえば for 〜ば) don't.
Particles have no markers because they appear in almost every sentence; they're found
through BM25 and embeddings instead.
