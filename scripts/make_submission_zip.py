"""Build the submission ZIP. Run from anywhere: python scripts/make_submission_zip.py [name.zip]"""

from __future__ import annotations

import fnmatch
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "Career_Copilot_A3.zip"

# Anything matching these (relative, posix paths) is left out.
EXCLUDE = (
    ".git/*", ".git", ".env", ".env.*", "venv/*", ".venv/*", "*/__pycache__/*", "__pycache__/*",
    "*.pyc", ".pytest_cache/*", "node_modules/*", "docs/chat_export.md", "_pdf_preview/*",
    # A1 materials must not be in the A3 ZIP
    "docs/A1_*", "A1_Report_Materials*", "*.zip",
    # a dry run is not a result
    "eval/results/*dry_run*",
)
REQUIRED = ("streamlit_app.py", "requirements.txt", "README.md", "eval/run_eval.py", "config.example")


def included_files(root: Path = ROOT) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in EXCLUDE):
            continue
        files.append(path)
    return files


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NAME
    files = included_files()
    rels = {p.relative_to(ROOT).as_posix() for p in files}
    missing = [r for r in REQUIRED if r not in rels]
    if missing:
        print(f"Missing required files: {missing}", file=sys.stderr)
        return 1
    out = ROOT / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    print(f"Wrote {out} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
