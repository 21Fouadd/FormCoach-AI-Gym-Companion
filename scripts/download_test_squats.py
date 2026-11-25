"""Downloads publicly available open-source squat test MP4 videos into test_videos/."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "test_videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPOS = [
    "namnq/workout-free-videos",
    "rohanx01/Squat-Analysis-Model",
    "rishic3/DepthCheck",
    "NgoQuocBao1010/Exercise-Correction",
    "ai-trainer-samples/squat",
]


def fetch_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "GRADPROJECT-Downloader"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def download_file(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "GRADPROJECT-Downloader"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if len(data) < 25_000:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def main() -> None:
    downloaded = []
    for repo in REPOS:
        tree = fetch_json(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
        if not tree:
            continue
        for item in tree.get("tree", []):
            p = item.get("path", "")
            low = p.lower()
            if not low.endswith(".mp4"):
                continue
            if "squat" not in low:
                continue
            if any(skip in low for skip in ("output", "annotated", "result", "processed")):
                continue
            safe_name = f"{repo.split('/')[0]}_{Path(p).name}"
            dest = OUT_DIR / safe_name
            if dest.exists() and dest.stat().st_size > 25_000:
                downloaded.append(dest)
                continue
            raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.request.quote(p)}"
            if download_file(raw_url, dest):
                print(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
                downloaded.append(dest)
            if len(downloaded) >= 6:
                break
        if len(downloaded) >= 6:
            break

    print(f"\nTotal squat test videos saved in {OUT_DIR}: {len(downloaded)}")
    for d in downloaded:
        print(" -", d)


if __name__ == "__main__":
    main()
