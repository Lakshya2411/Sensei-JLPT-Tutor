"""Core tests. Run from backend/:  python -m pytest -q   (or: python -m unittest discover tests)

These avoid network calls: the LLM HTTP layer is replaced with fakes.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "scripts"))

from app import nlp, prompts  # noqa: E402
from app.bm25 import BM25  # noqa: E402
from app.cache import ResponseCache  # noqa: E402
from app.llm import LLMClient, LLMError, Provider, parse_json  # noqa: E402
from app.ratelimit import RateLimiter  # noqa: E402
from app.retriever import GrammarRetriever  # noqa: E402

GRAMMAR = json.loads((ROOT / "data/grammar/n5_n4_grammar.json").read_text(encoding="utf-8"))


class TestNLP(unittest.TestCase):
    def test_kata_to_hira(self):
        self.assertEqual(nlp.kata_to_hira("タベル"), "たべる")
        self.assertEqual(nlp.kata_to_hira("コーヒー"), "こーひー")

    def test_normalize_fullwidth(self):
        self.assertEqual(nlp.normalize("ＡＢＣ１２３"), "ABC123")

    def test_search_terms_has_bigrams(self):
        terms = nlp.search_terms("食べている")
        self.assertIn("てい", terms)
        self.assertIn("いる", terms)

    def test_has_japanese(self):
        self.assertTrue(nlp.has_japanese("hello 日本"))
        self.assertFalse(nlp.has_japanese("hello"))

    @unittest.skipUnless(nlp.sudachi_available(), "SudachiPy not installed")
    def test_sudachi_tokens(self):
        tokens = nlp.analyze("本を読んでいます")
        self.assertEqual(tokens[0].surface, "本")
        self.assertEqual(tokens[0].reading, "ほん")
        self.assertIn("読む", [t.base for t in tokens])


class TestBM25(unittest.TestCase):
    def test_ranks_matching_doc_first(self):
        bm = BM25([["cat", "sat"], ["dog", "ran"], ["cat", "cat", "ran"]])
        scores = bm.scores(["cat"])
        self.assertEqual(scores[1], 0.0)
        self.assertGreater(scores[2], scores[0])  # more occurrences of 'cat'

    def test_empty_corpus(self):
        self.assertEqual(BM25([]).scores(["x"]), [])


class TestGrammarData(unittest.TestCase):
    def test_ids_unique_and_fields_present(self):
        ids = [g["id"] for g in GRAMMAR]
        self.assertEqual(len(ids), len(set(ids)))
        for g in GRAMMAR:
            for key in ("pattern", "level", "meaning", "formation", "explanation", "markers", "examples"):
                self.assertIn(key, g, g["id"])

    def test_every_example_matches_its_own_markers(self):
        r = GrammarRetriever(GRAMMAR)
        for g in GRAMMAR:
            if not g["markers"]:
                continue  # particles: too common for regex markers, found via BM25/embeddings
            for ex in g["examples"]:
                self.assertIn(g["id"], r.marker_ranking(ex["ja"]), ex["ja"])

    def test_known_false_positives_avoided(self):
        r = GrammarRetriever(GRAMMAR)
        self.assertNotIn("ba", r.marker_ranking("たとえば、すしが好きです。"))
        self.assertNotIn("ba", r.marker_ranking("行かなければなりません。"))
        self.assertNotIn("sou-appearance", r.marker_ranking("はい、そうです。"))


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.r = GrammarRetriever(GRAMMAR)

    def test_detects_pattern_in_sentence(self):
        ids = [h.grammar["id"] for h in self.r.detect("音楽を聞きながら走ります。")]
        self.assertIn("nagara", ids)

    def test_english_query(self):
        top = [h.grammar["id"] for h in self.r.search("have you ever done something before", k=3)]
        self.assertIn("ta-koto-ga-aru", top)

    def test_particle_question_finds_particle_note(self):
        top = [h.grammar["id"] for h in self.r.search("should I use を or に with 行く", k=3)]
        self.assertIn("particle-ni-he", top)

    def test_level_filter(self):
        hits = self.r.search("練習すれば上手になります", k=10, max_level="N5")
        self.assertTrue(all(h.grammar["level"] == "N5" for h in hits))

    def test_mined_examples_are_indexed(self):
        examples = [{"grammar_id": "nagara", "ja": "テレビを見ながらご飯を食べた。", "en": "x"}]
        r = GrammarRetriever(GRAMMAR, examples)
        self.assertEqual(len(r.examples), 1)
        hit = next(h for h in r.search("テレビを見ながら", k=3) if h.grammar["id"] == "nagara")
        self.assertEqual(hit.examples[0]["ja"], examples[0]["ja"])

    def test_embedding_failure_is_ignored(self):
        class Broken:
            def embed(self, texts):
                raise RuntimeError("quota")

        import numpy as np

        r = GrammarRetriever(GRAMMAR, embedder=Broken())
        r.grammar_vectors = np.ones((len(GRAMMAR), 4))
        self.assertTrue(r.search("ながら", k=3))  # still answers with BM25 + markers


class TestLLM(unittest.TestCase):
    def _providers(self):
        return [Provider("gemini", "https://g", "k1", "m1"), Provider("groq", "https://q", "k2", "m2")]

    def test_parse_json_with_fences(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('Sure! {"a": 2} hope this helps'), {"a": 2})

    def test_falls_back_to_groq_on_rate_limit(self):
        calls = []

        def fake_post(url, headers, payload, timeout):
            calls.append(url)
            if url.startswith("https://g"):
                raise LLMError("HTTP 429: quota exceeded")
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        client = LLMClient(self._providers(), http_post=fake_post)
        result, provider = client.chat_json("sys", "user")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(provider, "groq")
        self.assertEqual(len(calls), 2)  # no retry on 429, straight to fallback

    def test_all_fail_raises(self):
        def fake_post(*a):
            raise LLMError("HTTP 500: boom")

        client = LLMClient(self._providers(), http_post=fake_post, retries_per_provider=0)
        with self.assertRaises(LLMError):
            client.chat_json("s", "u")

    def test_no_keys(self):
        self.assertFalse(LLMClient([]).available)
        with self.assertRaises(LLMError):
            LLMClient([]).chat_json("s", "u")


class TestCacheAndLimits(unittest.TestCase):
    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            c = ResponseCache(Path(d) / "c.sqlite3")
            key = ResponseCache.make_key("explain", {"text": "猫"})
            self.assertIsNone(c.get(key))
            c.set(key, {"translation": "cat"})
            self.assertEqual(c.get(key), {"translation": "cat"})

    def test_rate_limiter(self):
        rl = RateLimiter(2, window_seconds=60)
        self.assertTrue(rl.allow("ip", now=0))
        self.assertTrue(rl.allow("ip", now=1))
        self.assertFalse(rl.allow("ip", now=2))
        self.assertTrue(rl.allow("other", now=2))
        self.assertTrue(rl.allow("ip", now=61))


class TestPrompts(unittest.TestCase):
    def test_hallucinated_ids_are_nulled(self):
        items = [{"id": "nagara"}, {"id": "made-up"}]
        out = prompts.keep_known_ids(items, {"nagara"})
        self.assertEqual([i["id"] for i in out], ["nagara", None])

    def test_explain_prompt_contains_notes(self):
        hit = GrammarRetriever(GRAMMAR).search("ながら", k=1)[0].to_dict()
        _, user = prompts.explain_prompt("歩きながら", [], [hit])
        self.assertIn("[nagara]", user)


class TestBuildScripts(unittest.TestCase):
    def test_mine_examples(self):
        from build_index import mine_examples

        pairs = [
            {"ja": "歌いながら料理する。", "en": "Cook while singing.", "attribution": "t"},
            {"ja": "これはペンです。", "en": "This is a pen.", "attribution": "t"},
        ]
        out = mine_examples(GRAMMAR, pairs)
        self.assertEqual([(e["grammar_id"], e["ja"]) for e in out], [("nagara", "歌いながら料理する。")])

    def test_build_dictionary(self):
        from build_index import build_dictionary

        xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE JMdict [<!ENTITY uk "word usually written using kana alone">]>
<JMdict>
<entry><k_ele><keb>猫</keb><ke_pri>ichi1</ke_pri></k_ele><r_ele><reb>ねこ</reb></r_ele>
<sense><gloss>cat</gloss></sense><sense><gloss>shamisen</gloss></sense></entry>
<entry><k_ele><keb>羽</keb><ke_pri>ichi1</ke_pri></k_ele><r_ele><reb>は</reb><re_pri>ichi1</re_pri></r_ele>
<sense><gloss>feather</gloss></sense></entry>
<entry><k_ele><keb>彼</keb><ke_pri>ichi1</ke_pri></k_ele><r_ele><reb>あれ</reb></r_ele>
<sense><misc>&uk;</misc><gloss>that</gloss></sense></entry>
<entry><k_ele><keb>彼</keb><ke_pri>ichi1</ke_pri></k_ele><r_ele><reb>かれ</reb></r_ele>
<sense><gloss>he</gloss><gloss>him</gloss></sense></entry>
<entry><k_ele><keb>珍語</keb></k_ele><r_ele><reb>ちんご</reb></r_ele><sense><gloss>rare word</gloss></sense></entry>
</JMdict>"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "jm.xml"
            p.write_text(xml, encoding="utf-8")
            data = build_dictionary(p)
        words = data["words"]
        self.assertEqual(words["猫"][0]["g"], ["cat"])  # first sense only
        self.assertNotIn("ねこ", words)  # kana spelling of a kanji word isn't indexed
        self.assertNotIn("は", words)  # so は can't become "feather"
        self.assertIn("あれ", words)  # usually-kana word is indexed by reading
        self.assertEqual(len(words["彼"]), 2)
        self.assertNotIn("珍語", words)  # not common, filtered out


class TestDictionary(unittest.TestCase):
    def setUp(self):
        from app.dictionary import Dictionary

        self.d = Dictionary({
            "彼": [{"r": ["あれ"], "g": ["that"]}, {"r": ["かれ"], "g": ["he"]}],
            "羽": [{"r": ["は"], "g": ["feather"]}],
            "友達": [{"r": ["ともだち"], "g": ["friend"]}],
        })

    def test_reading_picks_the_right_entry(self):
        self.assertEqual(self.d.lookup("彼", "彼", "カレ", "pronoun"), ["he"])
        self.assertEqual(self.d.lookup("彼", "彼", "あれ", "pronoun"), ["that"])

    def test_particles_use_role_table(self):
        self.assertEqual(self.d.lookup("は", "は", "わ", "particle"), ["topic marker"])
        self.assertEqual(self.d.lookup("です", "です", "です", "auxiliary"), ["is (polite)"])
        self.assertEqual(self.d.lookup("ます", "まし", "まし", "auxiliary"), ["polite ending"])

    def test_unknown_word(self):
        self.assertEqual(self.d.lookup("ぴよ", "ぴよ", "ぴよ", "noun"), [])


if __name__ == "__main__":
    unittest.main()
