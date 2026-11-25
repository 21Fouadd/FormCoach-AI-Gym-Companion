"""Verifies all FormCoach Dart files and creates formcoach_connected.zip."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FC_DIR = ROOT / "formcoach"
OUT_ZIP = ROOT / "formcoach_connected.zip"


def check_dart_file(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8")
    errs: list[str] = []
    if "package:material_ui" in txt:
        errs.append(f"{path.name}: still references material_ui")
    if "activeThumbColor" in txt:
        errs.append(f"{path.name}: still uses activeThumbColor")
    if "initialValue:" in txt and "DropdownButtonFormField" in txt:
        errs.append(f"{path.name}: still uses DropdownButtonFormField.initialValue")

    in_sq = in_dq = in_lc = in_bc = esc = False
    stack: list[str] = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    closers = {v: k for k, v in pairs.items()}
    i = 0
    while i < len(txt):
        ch = txt[i]
        nxt = txt[i + 1] if i + 1 < len(txt) else ""
        if in_lc:
            if ch == "\n":
                in_lc = False
        elif in_bc:
            if ch == "*" and nxt == "/":
                in_bc = False
                i += 1
        elif in_sq:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_sq = False
        elif in_dq:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_dq = False
        else:
            if ch == "/" and nxt == "/":
                in_lc = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_bc = True
                i += 1
            elif ch == "'":
                in_sq = True
            elif ch == '"':
                in_dq = True
            elif ch in pairs:
                stack.append(ch)
            elif ch in closers:
                if not stack or stack[-1] != closers[ch]:
                    errs.append(f"{path.name}: unmatched closer {ch} at idx {i}")
                    break
                stack.pop()
        i += 1
    if stack:
        errs.append(f"{path.name}: unclosed {stack[-1]}")
    return errs


def main() -> None:
    dart_files = sorted(FC_DIR.rglob("*.dart"))
    all_errs: list[str] = []
    for p in dart_files:
        all_errs.extend(check_dart_file(p))
    if all_errs:
        raise AssertionError("Dart verification failed:\n" + "\n".join(all_errs))
    print(f"Verified {len(dart_files)} Dart files cleanly (0 import/property/brace issues).")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(FC_DIR.rglob("*")):
            if f.is_file() and not any(part in (".dart_tool", "build", ".gradle") for part in f.parts):
                zf.write(f, arcname=f.relative_to(ROOT).as_posix())
    print(f"Created {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
