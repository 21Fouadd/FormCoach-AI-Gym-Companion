"""Checks Exercise-Correction and related GitHub repos + HuggingFace for bad-form squat MP4s."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "test_videos" / "bad_form"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "GRADPROJECT-BadSquatFinder"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def download_file(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "GRADPROJECT-BadSquatFinder"})
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
    repos = [
        "Exercise-Correction/core",
        "NgoQuocBao1010/Exercise-Correction",
        "Fares-A/AI-Fitness-Trainer",
        "AI-Fitness-Trainer/AI-Fitness-Trainer",
    ]
    for q in ("squat incorrect form", "ai fitness trainer squat", "squat posture correction"):
        res = fetch_json(f"https://api.github.com/search/repositories?q={urllib.parse.quote(q)}&per_page=10")
        if res:
            for item in res.get("items", []):
                fn = item.get("full_name")
                if fn and fn not in repos:
                    repos.append(fn)

    found = []
    for repo in repos:
        tree = fetch_json(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
        if not tree:
            continue
        for item in tree.get("tree", []):
            p = item.get("path", "")
            low = p.lower()
            if not low.endswith(".mp4"):
                continue
            if any(k in low for k in ("bad", "wrong", "incorrect", "poor", "fault", "shallow", "mistake", "err")):
                raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(p)}"
                dest = OUT_DIR / f"{repo.split('/')[0]}_{Path(p).name}"
                if download_file(raw_url, dest):
                    print(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024:.0f} KB) <- {repo}/{p}")
                    found.append(dest)
            if len(found) >= 4:
                break
        if len(found) >= 4:
            break

    # Fallback: download 2 real BAD-form side-view squat clips directly from Mendeley kgbb3yn47p (v3)
    if not found:
        print("No bad-form MP4s on GitHub; fetching 2 real side-view BAD squat clips from Mendeley kgbb3yn47p...")
        root_folders = fetch_json("https://data.mendeley.com/public-api/datasets/kgbb3yn47p/folders?version=3&folder_id=root") or []
        def walk_for_bad_squat(fid: str, prefix: str):
            if len(found) >= 2:
                return
            files = fetch_json(f"https://data.mendeley.com/public-api/datasets/kgbb3yn47p/files?version=3&folder_id={fid}") or []
            for f in files:
                fn = f.get("filename", "")
                full_p = f"{prefix}/{fn}".lower()
                if fn.lower().endswith(".mp4") and "squat" in full_p and "bad" in full_p and "side" in full_p:
                    url = (f.get("content_details") or {}).get("download_url")
                    if url:
                        dest = OUT_DIR / f"bad_squat_side_{fn}"
                        if download_file(url, dest):
                            print(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
                            found.append(dest)
                            if len(found) >= 2:
                                return
            subfolders = fetch_json(f"https://data.mendeley.com/public-api/datasets/kgbb3yn47p/folders?version=3&folder_id={fid}") or []
            for sf in subfolders:
                nm = (sf.get("name") or "").lower()
                if any(other in nm for other in ("push", "curl", "shoulder", "row", "tricep", "side_bend", "side bend")):
                    continue
                if "subject" in nm:
                    continue
                walk_for_bad_squat(sf["id"], f"{prefix}/{sf.get('name', '')}")
        for rf in root_folders:
            if "exercise" in (rf.get("name") or "").lower():
                walk_for_bad_squat(rf["id"], rf.get("name", ""))
                break

    print(f"\nSaved {len(found)} bad-form squat videos in {OUT_DIR}:")
    for f in found:
        print(" -", f)


if __name__ == "__main__":
    main()
