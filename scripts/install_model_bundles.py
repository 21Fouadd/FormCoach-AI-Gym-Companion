"""Installs exported Colab model ZIPs (`gym_model_bundle_remaining5.zip` or per-exercise zips) into backend/models/."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "backend" / "models"
DOWNLOADS = Path.home() / "Downloads"
EXERCISES = ("squat", "push", "curl", "shoulder", "row", "lunge", "arm_abduction")


def install_zip(zip_path: Path) -> None:
    print(f"Extracting {zip_path.name} into {MODELS_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        has_subdirs = any(n.split("/")[0] in EXERCISES for n in names if "/" in n)
        if has_subdirs:
            for member in names:
                top = member.split("/")[0]
                if top in EXERCISES and not member.endswith("/"):
                    target = MODELS_DIR / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
                    print(f"  Installed {target.relative_to(ROOT)}")
        else:
            ex = next((e for e in EXERCISES if e in zip_path.name.lower()), None)
            if ex is None:
                raise RuntimeError(f"Could not infer exercise name from {zip_path.name}")
            ex_dir = MODELS_DIR / ex
            ex_dir.mkdir(parents=True, exist_ok=True)
            for member in names:
                if not member.endswith("/"):
                    target = ex_dir / Path(member).name
                    target.write_bytes(zf.read(member))
                    print(f"  Installed {target.relative_to(ROOT)}")


def main() -> None:
    if len(sys.argv) > 1:
        zips = [Path(a) for a in sys.argv[1:]]
    else:
        zips = sorted(
            list(DOWNLOADS.glob("gym_model_bundle_push_shoulder_ensemble*.zip"))
            + list(DOWNLOADS.glob("gym_model_bundle_unified_ensemble*.zip"))
            + list(DOWNLOADS.glob("gym_model_bundle_curl_upgraded*.zip"))
            + list(DOWNLOADS.glob("gym_model_bundle_upgraded*.zip"))
            + list(DOWNLOADS.glob("gym_model_bundle_remaining5*.zip")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:1]
    if not zips:
        print(f"No `gym_model_bundle_remaining5.zip` found in {DOWNLOADS}. Download it from Colab first!")
        sys.exit(1)
    for zp in zips:
        install_zip(zp)

    sys.path.insert(0, str(ROOT / "backend"))
    from app.pose_engine import ModelRegistry

    summary = ModelRegistry().summary()
    print("\nLoaded Models Summary (ALL 7 EXERCISES):")
    for ex, mode in summary.items():
        print(f"  - {ex:14s}: {mode}")


if __name__ == "__main__":
    main()
