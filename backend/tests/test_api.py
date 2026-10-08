"""API smoke tests with a fake LLM. Run from backend/:  python -m pytest -q"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi.testclient import TestClient

    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import main
        from app.cache import ResponseCache
        from app.llm import LLMClient, Provider

        def fake_post(url, headers, payload, timeout):
            user = payload["messages"][1]["content"]
            if "practice questions" in user:
                content = {"questions": [{"type": "translate", "question": "q", "options": [], "answer": "a", "explanation": "e"}]}
            elif "learner wrote" in user:
                content = {"is_correct": False, "corrected": "x", "translation": "t",
                           "errors": [{"original": "a", "fix": "b", "explanation": "c", "grammar_id": "invented"}],
                           "more_natural": ""}
            else:
                content = {"translation": "I study while listening to music.", "literal": "", "words": [],
                           "grammar": [{"id": "nagara", "pattern": "ながら", "how_it_is_used_here": "..."},
                                       {"id": "invented-id", "pattern": "?", "how_it_is_used_here": "?"}],
                           "tip": ""}
            return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}

        fake = LLMClient([Provider("gemini", "https://fake", "k", "m")], http_post=fake_post)
        main.get_llm.cache_clear()
        main.get_llm = lambda: fake  # routes look up get_llm at call time
        cls.tmp = tempfile.TemporaryDirectory()
        main.get_cache.cache_clear()
        cache = ResponseCache(Path(cls.tmp.name) / "c.sqlite3")
        main.get_cache = lambda: cache
        main.limiter.max = 1000
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.json()["grammar_points"], 20)

    def test_analyze_detects_grammar(self):
        r = self.client.post("/analyze", json={"text": "音楽を聞きながら勉強します。"})
        ids = [g["id"] for g in r.json()["grammar"]]
        self.assertIn("nagara", ids)

    def test_explain_drops_invented_ids_and_caches(self):
        body = {"text": "音楽を聞きながら勉強します。"}
        first = self.client.post("/explain", json=body).json()
        ids = [g["id"] for g in first["answer"]["grammar"]]
        self.assertEqual(ids, ["nagara", None])
        self.assertFalse(first["answer"]["cached"])
        self.assertTrue(self.client.post("/explain", json=body).json()["answer"]["cached"])

    def test_explain_rejects_english(self):
        self.assertEqual(self.client.post("/explain", json={"text": "hello"}).status_code, 400)

    def test_practice(self):
        r = self.client.post("/practice", json={"grammar_id": "nagara", "n": 1})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["answer"]["questions"]), 1)
        self.assertEqual(self.client.post("/practice", json={"grammar_id": "nope"}).status_code, 404)

    def test_correct(self):
        r = self.client.post("/correct", json={"text": "私は学校を行きます。"}).json()
        self.assertIsNone(r["answer"]["errors"][0]["grammar_id"])


if __name__ == "__main__":
    unittest.main()
