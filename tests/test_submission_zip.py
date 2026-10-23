"""The submission ZIP must leave out secrets and A1 material."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_zip", Path(__file__).resolve().parent.parent / "scripts" / "make_submission_zip.py"
)
make_zip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_zip)


def test_zip_excludes_a1_secrets_and_caches(tmp_path: Path) -> None:
    for rel in ("streamlit_app.py", "requirements.txt", "README.md", "config.example",
                "eval/run_eval.py", ".env", "docs/A1_Report_1.pdf", "docs/A1_Report_1.md",
                "venv/lib/x.py", "app/__pycache__/a.pyc", "eval/results/results_dry_run.md",
                "eval/results/results.md", "old.zip", "docs/notes.md"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    rels = {p.relative_to(tmp_path).as_posix() for p in make_zip.included_files(tmp_path)}
    assert {"eval/results/results.md", "docs/notes.md", "eval/run_eval.py"} <= rels
    assert not any(x in rels for x in (".env", "docs/A1_Report_1.pdf", "docs/A1_Report_1.md",
                                       "venv/lib/x.py", "app/__pycache__/a.pyc",
                                       "eval/results/results_dry_run.md", "old.zip"))
