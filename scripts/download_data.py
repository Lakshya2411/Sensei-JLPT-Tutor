"""Download the open datasets into data/raw/.

Usage (from the project root):
    python scripts/download_data.py            # everything
    python scripts/download_data.py tatoeba    # just one

Sources and licenses (see data/README.md):
- Tatoeba Japanese-English pairs, via ManyThings.org's tab-separated export (CC BY 2.0 FR)
- JMdict English edition from EDRDG (CC BY-SA 4.0)

If a URL has moved, download the file manually and put it in data/raw/ with
the filename shown below; the build script only cares about the filename.
"""
from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "tatoeba": {
        "url": "https://www.manythings.org/anki/jpn-eng.zip",
        "archive": "jpn-eng.zip",
        "final": "jpn.txt",
    },
    "jmdict": {
        "url": "http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz",
        "archive": "JMdict_e.gz",
        "final": "JMdict_e.xml",
    },
}


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "sensei-jlpt-tutor/0.1 (educational project)"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def extract(name: str, archive: Path, final: Path) -> None:
    if name == "tatoeba":
        with zipfile.ZipFile(archive) as z:
            member = next(m for m in z.namelist() if m.endswith(".txt") and "_about" not in m)
            with z.open(member) as src, final.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    elif name == "jmdict":
        with gzip.open(archive, "rb") as src, final.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"  extracted -> data/raw/{final.name}")


def main(names: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in names or SOURCES:
        src = SOURCES[name]
        final = RAW / src["final"]
        if final.exists():
            print(f"{final.name} already exists, skipping")
            continue
        archive = RAW / src["archive"]
        if not archive.exists():
            download(src["url"], archive)
        extract(name, archive, final)


if __name__ == "__main__":
    main(sys.argv[1:])
