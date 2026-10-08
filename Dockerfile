# Backend image. Works on Hugging Face Spaces (Docker SDK), Render and Cloud Run.
# Build from the project root:  docker build -t sensei .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Knowledge base is built ahead of time (scripts/build_index.py) and shipped in the image.
COPY data/grammar data/grammar
COPY data/processed data/processed
COPY backend backend

# HF Spaces runs containers as a non-root user; the cache file must be writable.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

WORKDIR /app/backend
ENV CACHE_PATH=/tmp/cache.sqlite3
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
