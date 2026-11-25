"""Downloads publicly available open-source push-up test MP4 videos into test_videos/pushups/."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "test_videos" / "pushups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_REPOS = [
    "namnq/workout-free-videos",
]
MATCH_KEYWORDS = ("push-up", "push_up", "pushup")
SKIP_KEYWORDS = ("output", "annotated", "result", "processed")


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


def _collect_from_repo(repo: str, downloaded: list[Path], limit: int = 6) -> None:
    tree = fetch_json(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
    if not tree:
        return
    for item in tree.get("tree", []):
        if len(downloaded) >= limit:
            break
        p = item.get("path", "")
        low = p.lower()
        if not low.endswith(".mp4"):
            continue
        if not any(kw in low for kw in MATCH_KEYWORDS):
            continue
        if any(skip in low for skip in SKIP_KEYWORDS):
            continue
        safe_name = f"{repo.split('/')[0]}_{Path(p).name}"
        dest = OUT_DIR / safe_name
        if dest in downloaded:
            continue
        if dest.exists() and dest.stat().st_size > 25_000:
            print(f"Already cached: {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
            downloaded.append(dest)
            continue
        raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(p)}"
        if download_file(raw_url, dest):
            print(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
            downloaded.append(dest)


def main() -> None:
    downloaded: list[Path] = []
    seen_repos: set[str] = set()

    for repo in SEED_REPOS:
        seen_repos.add(repo)
        _collect_from_repo(repo, downloaded, limit=6)

    if len(downloaded) < 6:
        search_url = (
            "https://api.github.com/search/repositories?q="
            + urllib.parse.quote("pushup exercise mediapipe")
            + "&per_page=15"
        )
        res = fetch_json(search_url)
        if res:
            for item in res.get("items", []):
                if len(downloaded) >= 6:
                    break
                full_name = item.get("full_name")
                if full_name and full_name not in seen_repos:
                    seen_repos.add(full_name)
                    _collect_from_repo(full_name, downloaded, limit=6)

    print(f"\nTotal push-up test videos saved in {OUT_DIR}: {len(downloaded)}")
    for d in downloaded:
        print(f" - {d} ({d.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
