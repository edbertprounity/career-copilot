"""Cross-user isolation: Maya vs Liam. 0 Gemini calls."""

from __future__ import annotations

import ast
import json
import re
import tempfile
import zlib
from pathlib import Path

import app.cv_generator as cv_generator
import app.llm_client as llm_client
import app.vector_store as vector_store
import streamlit_app as ui
from app.agent_engine import finalize_and_confirm_profile, inspect_completeness
from app.cv_generator import (
    _templated_summary,
    generate_cv_markdown,
    generate_cv_pdf,
)
from app.schemas import StudentProfile
from app.vector_store import search_top_jobs
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "data" / "student_profiles.jsonl"
APP_PATH = ROOT / "streamlit_app.py"
APP_MODULES = (
    ROOT / "app" / "agent_engine.py",
    ROOT / "app" / "vector_store.py",
    ROOT / "app" / "cv_generator.py",
)
SHARED_CACHE_NAMES = frozenset({"_jobs", "_embeddings", "_model"})


def _load_row(document_id: str) -> dict:
    for line in PROFILES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("document_id") == document_id:
            return row
    raise AssertionError(f"{document_id} missing from student_profiles.jsonl")


def _profile(document_id: str) -> StudentProfile:
    return StudentProfile.model_validate(_load_row(document_id))


def _identity_needles(owner: StudentProfile, other: StudentProfile) -> list[str]:
    """Markers that identify a person, not shared job-market vocabulary."""
    needles = [owner.name]
    other_titles = {item.title for item in other.projects}
    for item in owner.projects:
        if item.title and item.title not in other_titles:
            needles.append(item.title)
    other_roles = {item.role for item in other.experience}
    other_orgs = {item.organisation for item in other.experience}
    for item in owner.experience:
        if item.role and item.role not in other_roles:
            needles.append(item.role)
        if item.organisation and item.organisation not in other_orgs:
            needles.append(item.organisation)
    owner_github = (owner.links or {}).get("github") or ""
    other_github = (other.links or {}).get("github") or ""
    if owner_github and owner_github != other_github:
        needles.append(owner_github.replace("https://", "").replace("www.", ""))
    return [item for item in needles if item.strip()]


def _exclusive_cv_needles(owner: StudentProfile, other: StudentProfile) -> list[str]:
    needles = list(_identity_needles(owner, other))
    other_targets = {role.lower() for role in other.target_roles}
    for role in owner.target_roles:
        if role.lower() not in other_targets:
            needles.append(role)
    return needles


def _blob(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, default=str).lower()


def _assert_no_foreign_identity(
    label: str,
    payload: object,
    owner: StudentProfile,
    other: StudentProfile,
) -> None:
    text = _blob(payload)
    for needle in _identity_needles(other, owner):
        assert needle.lower() not in text, (
            f"{label} leaked other-user marker {needle!r}"
        )


def _assert_owner_cv(
    label: str,
    payload: object,
    owner: StudentProfile,
    other: StudentProfile,
) -> None:
    text = _blob(payload)
    for needle in _identity_needles(owner, other):
        assert needle.lower() in text, f"{label} missing owner marker {needle!r}"
    for needle in _exclusive_cv_needles(other, owner):
        assert needle.lower() not in text, (
            f"{label} leaked other-user marker {needle!r}"
        )


def _pdf_text(pdf_bytes: bytes) -> str:
    chunks = [pdf_bytes.decode("latin-1", errors="ignore")]
    for match in re.finditer(rb"stream\r?\n(.+?)\r?\nendstream", pdf_bytes, re.S):
        raw = match.group(1)
        try:
            chunks.append(zlib.decompress(raw).decode("latin-1", errors="ignore"))
        except zlib.error:
            chunks.append(raw.decode("latin-1", errors="ignore"))
    return "\n".join(chunks)


def _job_description(document_id: str) -> str:
    for job in vector_store.load_jobs():
        if job.document_id == document_id:
            return job.description
    return ""


def _cv_for(profile: StudentProfile, match: dict) -> str:
    description = _job_description(str(match.get("document_id") or ""))
    title = str(match.get("title") or "")
    summary = _templated_summary(profile, description, title)
    return generate_cv_markdown(
        profile, match, description, professional_summary=summary
    )


def test_pipeline_isolation_maya_and_liam() -> None:
    maya = _profile("student_profile_01")
    liam = _profile("student_profile_02")
    assert maya.name == "Maya Chen"
    assert liam.name == "Liam Patel"
    assert maya is not liam
    assert maya.projects is not liam.projects

    maya_inspect = inspect_completeness(maya)
    liam_inspect = inspect_completeness(liam)
    assert maya_inspect["complete"] is True
    assert liam_inspect["complete"] is False
    assert "experience" in liam_inspect["missing_fields"]
    assert inspect_completeness(maya)["complete"] is True
    assert liam.name == "Liam Patel"
    assert maya.name == "Maya Chen"

    maya_finalize = finalize_and_confirm_profile(maya)
    liam_finalize = finalize_and_confirm_profile(liam)
    assert maya_finalize["status"] == "success"
    assert maya.is_confirmed is True
    assert liam_finalize["status"] == "error"
    assert liam.is_confirmed is False
    assert "experience" in (liam_finalize.get("missing_fields") or [])

    # Skip-path equivalent: confirm Liam without copying Maya's experience.
    liam.is_confirmed = True
    assert not liam.experience
    assert any(item.role == "Customer Service Assistant" for item in maya.experience)

    maya_matches = search_top_jobs(maya, top_k=3)
    liam_matches = search_top_jobs(liam, top_k=3)
    assert maya_matches is not liam_matches
    assert maya_matches[0] is not liam_matches[0]
    maya_matches[0]["title"] = "MUTATED MAYA JOB TITLE"
    assert liam_matches[0].get("title") != "MUTATED MAYA JOB TITLE"
    _assert_no_foreign_identity("maya matches", maya_matches, maya, liam)
    _assert_no_foreign_identity("liam matches", liam_matches, liam, maya)
    for match in maya_matches + liam_matches:
        excerpt = str(match.get("supporting_excerpt") or "").lower()
        assert "maya chen" not in excerpt
        assert "liam patel" not in excerpt

    maya_md = _cv_for(maya, maya_matches[0])
    liam_md = _cv_for(liam, liam_matches[0])
    snapshot_maya = maya_md
    _assert_owner_cv("maya markdown", maya_md, maya, liam)
    _assert_owner_cv("liam markdown", liam_md, liam, maya)
    assert "Target Roles" not in maya_md
    assert "Target Roles" not in liam_md
    assert snapshot_maya == maya_md

    with tempfile.TemporaryDirectory() as tmp:
        maya_pdf_path = Path(tmp) / "maya.pdf"
        liam_pdf_path = Path(tmp) / "liam.pdf"
        generate_cv_pdf(maya_md, str(maya_pdf_path))
        generate_cv_pdf(liam_md, str(liam_pdf_path))
        maya_pdf = _pdf_text(maya_pdf_path.read_bytes())
        liam_pdf = _pdf_text(liam_pdf_path.read_bytes())
    assert "Maya Chen" in maya_pdf
    assert "Liam Patel" in liam_pdf
    assert "Liam Patel" not in maya_pdf
    assert "Maya Chen" not in liam_pdf
    assert "Team Course Planner" in maya_pdf
    assert "Team Course Planner" not in liam_pdf
    assert "Text Classification Study" in liam_pdf
    assert "Text Classification Study" not in maya_pdf
    assert "Customer Service Assistant" in maya_pdf
    assert "Customer Service Assistant" not in liam_pdf


def test_interleaved_cv_calls_do_not_bleed() -> None:
    maya = _profile("student_profile_01")
    liam = _profile("student_profile_02")
    maya.is_confirmed = True
    liam.is_confirmed = True
    maya_match = search_top_jobs(maya, top_k=1)[0]
    liam_match = search_top_jobs(liam, top_k=1)[0]

    maya_md = _cv_for(maya, maya_match)
    liam_md = _cv_for(liam, liam_match)
    maya.name = "HACKED MAYA"
    liam.name = "HACKED LIAM"

    with tempfile.TemporaryDirectory() as tmp:
        maya_pdf_path = Path(tmp) / "maya.pdf"
        liam_pdf_path = Path(tmp) / "liam.pdf"
        generate_cv_pdf(maya_md, str(maya_pdf_path))
        generate_cv_pdf(liam_md, str(liam_pdf_path))
        maya_pdf = _pdf_text(maya_pdf_path.read_bytes())
        liam_pdf = _pdf_text(liam_pdf_path.read_bytes())

    assert "Maya Chen" in maya_pdf
    assert "Liam Patel" in liam_pdf
    assert "HACKED MAYA" not in maya_pdf
    assert "HACKED LIAM" not in liam_pdf
    assert "HACKED MAYA" not in liam_pdf
    assert "Liam Patel" not in maya_pdf
    assert "Maya Chen" not in liam_pdf


def test_no_profile_specific_module_level_cache() -> None:
    assigned: dict[str, set[str]] = {}
    globals_used: dict[str, set[str]] = {}
    for path in APP_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        used: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                used.update(node.names)
        assigned[path.name] = names
        globals_used[path.name] = used

    assert globals_used["agent_engine.py"] == set()
    assert globals_used["cv_generator.py"] == set()
    assert globals_used["vector_store.py"] <= SHARED_CACHE_NAMES
    assert assigned["vector_store.py"] & {"_jobs", "_embeddings", "_model"} == {
        "_jobs",
        "_embeddings",
        "_model",
    }

    maya = _profile("student_profile_01")
    liam = _profile("student_profile_02")
    maya.is_confirmed = True
    liam.is_confirmed = True
    search_top_jobs(maya, top_k=1)
    search_top_jobs(liam, top_k=1)
    assert getattr(vector_store, "_jobs", None) is not None
    job_blob = _blob(
        [
            {"title": job.title, "document_id": job.document_id}
            for job in vector_store._jobs
        ]
    )
    assert "maya chen" not in job_blob
    assert "liam patel" not in job_blob
    assert not hasattr(vector_store, "_last_profile")
    assert not hasattr(cv_generator, "_last_profile")
    assert not hasattr(llm_client, "_last_profile")


def test_skip_path_does_not_mutate_shared_catalog_rows() -> None:
    cards = {card["document_id"]: card for card in ui._demo_catalog()}
    row = cards["student_profile_01"]["profile_row"]
    before = json.dumps(row, sort_keys=True)
    profile = StudentProfile.model_validate(row)
    profile.is_confirmed = True
    profile.name = "MUTATED SESSION COPY"
    profile.technical_skills.append("SHOULD NOT WRITE BACK")
    after = json.dumps(cards["student_profile_01"]["profile_row"], sort_keys=True)
    assert before == after
    fresh = StudentProfile.model_validate(
        {card["document_id"]: card for card in ui._demo_catalog()}[
            "student_profile_01"
        ]["profile_row"]
    )
    assert fresh.name == "Maya Chen"
    assert "SHOULD NOT WRITE BACK" not in fresh.technical_skills


def test_two_streamlit_sessions_do_not_share_profiles() -> None:
    maya_at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    liam_at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    maya_at.session_state["pending_demo_action"] = "student_profile_01"
    liam_at.session_state["pending_demo_action"] = "student_profile_02"
    maya_at.run()
    liam_at.run()

    maya_profile = maya_at.session_state["profile"]
    liam_profile = liam_at.session_state["profile"]
    assert maya_profile.name == "Maya Chen"
    assert liam_profile.name == "Liam Patel"
    assert maya_profile is not liam_profile
    assert maya_at.session_state["profile"] is not liam_at.session_state["profile"]
    _assert_owner_cv(
        "maya session profile",
        maya_profile.model_dump(),
        maya_profile,
        liam_profile,
    )
    _assert_owner_cv(
        "liam session profile",
        liam_profile.model_dump(),
        liam_profile,
        maya_profile,
    )
    _assert_no_foreign_identity(
        "maya session jobs",
        maya_at.session_state["job_matches"],
        maya_profile,
        liam_profile,
    )
    _assert_no_foreign_identity(
        "liam session jobs",
        liam_at.session_state["job_matches"],
        liam_profile,
        maya_profile,
    )

    maya_at.session_state["profile"].name = "SESSION MUTATION MAYA"
    assert liam_at.session_state["profile"].name == "Liam Patel"
