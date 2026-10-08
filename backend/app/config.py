"""App settings, read from environment variables (and a .env file if python-dotenv is installed)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    allowed_origins: list = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5500").split(",")
            if o.strip()
        ]
    )
    rate_limit_per_min: int = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
    data_dir: Path = _path(os.getenv("DATA_DIR", "../data/processed"))
    grammar_file: Path = _path(os.getenv("GRAMMAR_FILE", "../data/grammar/n5_n4_grammar.json"))
    cache_path: Path = _path(os.getenv("CACHE_PATH", "./cache.sqlite3"))


settings = Settings()
