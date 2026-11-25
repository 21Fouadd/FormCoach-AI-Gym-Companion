"""Downloads real open-source test MP4 clips for Curl, Shoulder Press/OHP, Row/Deadlift, Lunge, and Arm Abduction."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

BASE_OUT = Path(__file__).resolve().parent.parent / "test_videos"

EXERCISE_SPECS = {
    "curls": {
        "match": ("bicep-curl", "biceps-curl", "dumbbell-curl", "barbell-curl", "hammer-curl", "curl"),
        "skip": ("leg-curl", "leg_curl", "hamstring", "nordic", "wrist-curl", "crunch", "output", "annotated"),
        "queries": ["bicep curl mediapipe", "bicep curl exercise video"],
    },
    "shoulder": {
        "match": ("shoulder-press", "overhead-press", "military-press", "dumbbell-shoulder-press", "arnold-press"),
        "skip": ("leg-press", "bench-press", "chest-press", "output", "annotated"),
        "queries": ["shoulder press mediapipe", "overhead press exercise video"],
    },
    "row": {
        "match": ("bent-over-row", "barbell-row", "dumbbell-row", "romanian-deadlift", "deadlift"),
        "skip": ("upright-row", "single-leg", "output", "annotated"),
        "queries": ["deadlift mediapipe", "bent over row mediapipe"],
    },
    "lunge": {
        "match": ("lunge", "forward-lunge", "reverse-lunge", "walking-lunge", "split-squat"),
        "skip": ("side-lunge", "lateral-lunge", "output", "annotated"),
        "queries": ["lunge exercise mediapipe"],
    },
    "arm_abduction": {
        "match": ("lateral-raise", "side-lateral-raise", "dumbbell-lateral-raise", "shoulder-abduction", "arm-abduction"),
        "skip": ("bent-over", "rear-delt", "lying", "cable-hip", "hip-abduction", "output", "annotated"),
        "queries": ["lateral raise mediapipe", "arm abduction exercise"],
    },
}


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
        if len(data) < 25_000 or len(data) > 25_000_000:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def collect_from_tree(repo: str, tree_items: list[dict], spec: dict, out_dir: Path, downloaded: list[Path], limit: int = 2) -> None:
    # Sort so exact primary matches appear first
    def priority(item: dict) -> int:
        low = item.get("path", "").lower()
        for idx, kw in enumerate(spec["match"]):
            if kw in low:
                return idx
        return 99

    candidates = [
        it for it in tree_items
        if it.get("path", "").lower().endswith(".mp4")
        and any(kw in it.get("path", "").lower() for kw in spec["match"])
        and not any(sk in it.get("path", "").lower() for sk in spec["skip"])
    ]
    candidates.sort(key=priority)

    for item in candidates:
        if len(downloaded) >= limit:
            break
        p = item["path"]
        safe_name = f"{repo.split('/')[0]}_{Path(p).name}"
        dest = out_dir / safe_name
        if dest in downloaded:
            continue
        if dest.exists() and dest.stat().st_size > 25_000:
            print(f"  [cached] {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
            downloaded.append(dest)
            continue
        raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(p)}"
        if download_file(raw_url, dest):
            print(f"  [downloaded] {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
            downloaded.append(dest)


def main() -> None:
    namnq_tree_resp = fetch_json("https://api.github.com/repos/namnq/workout-free-videos/git/trees/HEAD?recursive=1")
    namnq_items = namnq_tree_resp.get("tree", []) if namnq_tree_resp else []
    print(f"namnq/workout-free-videos tree entries: {len(namnq_items)}")

    for folder, spec in EXERCISE_SPECS.items():
        out_dir = BASE_OUT / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Collecting {folder} ===")
        downloaded: list[Path] = []
        if namnq_items:
            collect_from_tree("namnq/workout-free-videos", namnq_items, spec, out_dir, downloaded, limit=2)

        if len(downloaded) < 2:
            seen = {"namnq/workout-free-videos"}
            for q in spec["queries"]:
                if len(downloaded) >= 2:
                    break
                s_url = "https://api.github.com/search/repositories?q=" + urllib.parse.quote(q) + "&per_page=8"
                res = fetch_json(s_url)
                if not res:
                    continue
                for repo_item in res.get("items", []):
                    if len(downloaded) >= 2:
                        break
                    fn = repo_item.get("full_name")
                    if not fn or fn in seen:
                        continue
                    seen.add(fn)
                    tr = fetch_json(f"https://api.github.com/repos/{fn}/git/trees/HEAD?recursive=1")
                    if tr:
                        collect_from_tree(fn, tr.get("tree", []), spec, out_dir, downloaded, limit=2)
        print(f"Total for {folder}: {len(downloaded)}")


if __name__ == "__main__":
    main()
