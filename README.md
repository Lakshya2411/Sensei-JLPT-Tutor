# Sensei · JLPT tutor with grounded RAG

A web app for Japanese learners (JLPT N5–N4) that:

- **Explains** any Japanese sentence: furigana, word-by-word glosses and the grammar points it uses, citing a curated grammar knowledge base.
- **Generates practice** questions for a chosen grammar point, grounded in real example sentences.
- **Corrects** learners' sentences and explains each mistake.

Built to run entirely on free tiers.

## Live demo

**[sensei-jlpt-tutor-frontend.vercel.app](https://sensei-jlpt-tutor-frontend.vercel.app)**

> The backend runs on Render's free tier and sleeps when idle, so the first request can take about a minute. If the badge says "backend offline", wait a moment and refresh.

API health check: [sensei-jlpt-tutor.onrender.com/health](https://sensei-jlpt-tutor.onrender.com/health)

## Architecture

```
Static site (Vercel)  ──────────►  FastAPI backend (Render, Docker)  
                                         │
                                         ├─ SudachiPy: word segmentation, readings, dictionary forms
                                         ├─ JMdict lookup: English glosses (no LLM needed)
                                         ├─ Hybrid retrieval over grammar notes + Tatoeba examples
                                         │    ├─ regex pattern markers   (exact, high precision)
                                         │    ├─ BM25                    (keywords, hand-written)
                                         │    ├─ Gemini embeddings       (semantic, optional)
                                         │    └─ fused with Reciprocal Rank Fusion
                                         ├─ LLM: Gemini → Groq fallback (OpenAI-compatible APIs)
                                         ├─ Citation guard: drops grammar ids the LLM invents
                                         └─ SQLite cache + per-IP rate limit (protects free quota)
```

## Project layout

```
backend/app/      FastAPI app (main.py) and modules: nlp, bm25, retriever, llm, cache, ...
backend/tests/    Unit tests (no network) + API tests with a fake LLM
data/grammar/     Hand-written grammar notes (the knowledge base)
scripts/          download_data.py, build_index.py
eval/             Retrieval eval set + script (recall@k, MRR)
frontend/         Static site: index.html, app.js, style.css, config.js
Dockerfile        Backend image for Render (also works on Cloud Run)
```

## Run it locally

**1. Backend**

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

cp .env.example .env                 # then paste your free API keys into .env
#   GEMINI_API_KEY from https://aistudio.google.com  (do NOT enable billing)
#   GROQ_API_KEY   from https://console.groq.com

python scripts/download_data.py      # Tatoeba + JMdict into data/raw/
python scripts/build_index.py        # builds data/processed/
# optional, needs GEMINI_API_KEY:
python scripts/build_index.py --embed

cd backend
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs to try the API, and http://localhost:8000/health to check
what's loaded (SudachiPy, providers, example and dictionary counts).

**2. Frontend**

```bash
cd frontend
python -m http.server 5500
```

Open http://localhost:5500.

**3. Tests and eval**

```bash
cd backend && python -m pytest -q
cd .. && python eval/eval_retrieval.py --show-misses
```

## Deploy for free

The live demo runs on Render (backend) and Vercel (frontend), both on free plans.

**Backend → Render (Docker)**
1. Build the knowledge base locally (`download_data.py`, then `build_index.py`) and commit `data/processed/`.
2. On Render, create a **Web Service** from this repo. It detects the root `Dockerfile`; choose the **Free** instance type.
3. Add environment variables `GEMINI_API_KEY` and `GROQ_API_KEY`.
4. Deploy and open `<your-service>.onrender.com/health` to check that examples and dictionary words loaded.

**Frontend → Vercel**
1. Set `window.SENSEI_API` in `frontend/config.js` to your Render URL.
2. Import the repo on Vercel with **Root Directory** set to `frontend` and framework preset **Other** (no build step).
3. Back on Render, set `ALLOWED_ORIGINS` to your Vercel URL (exact match, no trailing slash) so the browser can call the API.

Free Render services sleep after about 15 minutes without traffic and wake on the next request.

## Evaluation

`eval/retrieval_questions.json` has Japanese sentences and English questions with the expected
grammar point. Current results with the seed data (no embeddings, character bigram fallback):

| Method | Recall@5 | MRR |
|---|---|---|
| Markers only | 0.48 | 0.44 |
| BM25 only | 1.00 | 0.96 |
| Hybrid (markers + BM25) | 1.00 | 0.96 |

These are optimistic: the questions were written alongside the notes. Add harder,
paraphrased questions (aim for 100) before quoting numbers on a resume.

## Roadmap

- [x] Week 1: knowledge base, SudachiPy pipeline, hybrid retrieval, API, frontend
- [ ] Grow grammar notes to 60–80 points; add JLPT vocab levels (Tanos lists)
- [ ] 100-question retrieval eval; add embeddings and compare
- [ ] Fine-tune a 1.5B–3B model with QLoRA (Unsloth on Kaggle/Colab) on Lang-8 for corrections
- [ ] Three-way comparison: prompting vs RAG vs fine-tuned (precision/recall/F0.5)
- [x] Deploy (Render + Vercel, free tiers)
- [ ] Record a demo GIF and add results to this README

## Credits

Example sentences: Tatoeba (CC BY 2.0 FR). Dictionary: JMdict, EDRDG (CC BY-SA 4.0).
