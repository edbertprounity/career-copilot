"""Deterministic CV generation tests. No live Gemini calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.cv_generator as cvg
from app.cv_generator import (
    LAYOUT_THRESHOLD,
    CvNotConfirmedError,
    _cv_skills,
    _description_terms,
    _ungrounded_summary_terms,
    cv_layout,
    cv_richness,
    generate_cv_markdown,
    generate_cv_summary,
)
from app.schemas import ExperienceRecord, ProjectRecord, StudentProfile

PROFILES_PATH = Path(__file__).resolve().parents[1] / "data" / "student_profiles.jsonl"


def _load_maya() -> StudentProfile:
    for line in PROFILES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("document_id") == "student_profile_01":
            profile = StudentProfile.model_validate(row)
            profile.is_confirmed = True
            return profile
    raise AssertionError("student_profile_01 not found")


def test_generate_cv_markdown_rejects_unconfirmed(
    complete_profile: StudentProfile,
) -> None:
    with pytest.raises(CvNotConfirmedError):
        generate_cv_markdown(complete_profile, {"document_id": "x"}, "a job")


def test_generate_cv_markdown_confirmed_has_professional_sections(
    confirmed_profile: StudentProfile,
) -> None:
    markdown = generate_cv_markdown(
        confirmed_profile, {"title": "AI Engineer"}, "Python and Flask"
    )
    assert "## Skills" in markdown
    assert "## Professional Summary" in markdown
    assert "Target Roles" not in markdown
    assert "Growth Areas" not in markdown


def test_description_terms_drop_sentence_initial_verbs() -> None:
    known = frozenset({"python", "rest apis", "docker"})
    mined = _description_terms(
        "Fixed small backend issues. Implemented Java endpoints. "
        "Worked in a small team. Connected a language model.",
        known,
    )
    lowered = {item.lower() for item in mined}
    assert "fixed" not in lowered
    assert "implemented" not in lowered
    assert "worked" not in lowered
    assert "connected" not in lowered
    assert "java" in lowered


def test_description_terms_keep_genuine_technical_terms() -> None:
    known = frozenset({"python", "rest apis"})
    mined = _description_terms(
        "REST APIs were exposed for the marking tool. PyTest covered the helpers.",
        known,
    )
    lowered = {item.lower() for item in mined}
    assert "rest" in lowered
    assert "apis" in lowered
    assert "pytest" in lowered


def test_cv_skills_include_curated_and_project_tech(
    confirmed_profile: StudentProfile,
) -> None:
    skills = {item.lower() for item in _cv_skills(confirmed_profile)}
    assert "python" in skills
    assert "sql" in skills
    assert "flask" in skills


def test_adaptive_layout_below_threshold_is_single() -> None:
    sparse = StudentProfile(
        name="Sparse",
        technical_skills=["Python"],
        projects=[ProjectRecord(title="One")],
        experience=[],
        is_confirmed=True,
    )
    assert cv_richness(sparse) < LAYOUT_THRESHOLD
    assert cv_layout(sparse) == "single"


def test_adaptive_layout_at_threshold_is_two_column() -> None:
    rich = StudentProfile(
        name="Rich",
        technical_skills=["a", "b", "c", "d"],
        projects=[ProjectRecord(title="p1"), ProjectRecord(title="p2")],
        experience=[
            ExperienceRecord(role="r1"),
            ExperienceRecord(role="r2"),
        ],
        is_confirmed=True,
    )
    assert cv_richness(rich) == LAYOUT_THRESHOLD
    assert cv_layout(rich) == "two-column"


def test_grounding_rejects_kubernetes_and_led_development() -> None:
    maya = _load_maya()
    flagged = _ungrounded_summary_terms(
        "Maya Chen led development of an enterprise Kubernetes platform.",
        maya,
    )
    assert "Kubernetes" in flagged
    assert "led development" in flagged


def test_grounding_accepts_profile_true_terms() -> None:
    maya = _load_maya()
    flagged = _ungrounded_summary_terms(
        "Maya Chen used Python and Flask on Team Course Planner.",
        maya,
    )
    assert flagged == []


def test_grounding_sentence_boundary_does_not_merge_pytest_her() -> None:
    maya = _load_maya()
    flagged = _ungrounded_summary_terms(
        "Maya Chen has experience with PyTest. Her projects include Team Course Planner.",
        maya,
    )
    assert flagged == []
    assert not any("her" in item.lower() and "pytest" in item.lower() for item in flagged)


def test_grounding_skips_sentence_initial_english() -> None:
    maya = _load_maya()
    skilled = _ungrounded_summary_terms(
        "Skilled in Python and SQL, Maya built Team Course Planner.",
        maya,
    )
    assert skilled == []
    with_k8s = _ungrounded_summary_terms(
        "With Docker and Kubernetes, she shipped a service.",
        maya,
    )
    assert not any(item.lower() == "with" for item in with_k8s)
    assert "Kubernetes" in with_k8s


def test_grounding_job_title_is_not_an_ungrounded_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming the target role is allowed; fabricated skills/experience still fail."""
    profile = StudentProfile(
        name="Liam Patel",
        degree="Bachelor of Computing Science",
        institution="University of Technology Sydney",
        year="Third year",
        technical_skills=["Python", "pandas", "NumPy"],
        projects=[
            ProjectRecord(
                title="Text Classification Study",
                description="Trained a baseline classifier and compared precision.",
            )
        ],
        target_roles=["AI Intern"],
        is_confirmed=True,
    )
    job_title = "Graduate AI Engineer"
    tailored = (
        "Liam Patel is tailored for the Graduate AI Engineer role and used "
        "Python and pandas on Text Classification Study."
    )
    assert _ungrounded_summary_terms(tailored, profile, job_title=job_title) == []

    fabricated = (
        "Liam Patel is tailored for the Graduate AI Engineer role and led "
        "development of an enterprise Kubernetes platform."
    )
    flagged = _ungrounded_summary_terms(fabricated, profile, job_title=job_title)
    assert "Kubernetes" in flagged
    assert "led development" in flagged
    assert not any("graduate" in item.lower() for item in flagged)
    assert not any("ai engineer" in item.lower() for item in flagged)

    def stub_call(_client, **_kwargs):
        return {"summary": tailored}

    monkeypatch.setattr(cvg, "call_gemini_json", stub_call)
    diagnostics: dict = {}
    text = generate_cv_summary(
        object(),
        profile,
        "Python pandas NumPy graduate role",
        job_title,
        diagnostics=diagnostics,
    )
    assert diagnostics["path"] == "gemini"
    assert text == tailored


def test_grounding_rejects_target_role_framed_as_held_experience() -> None:
    profile = StudentProfile(
        name="Sofia Williams",
        degree="Bachelor of Computing Science",
        institution="University of Technology Sydney",
        year="Third year",
        technical_skills=["Python", "SQL", "pandas"],
        projects=[
            ProjectRecord(
                title="Student Services Dashboard",
                description="Cleaned survey data and built an interactive dashboard.",
            )
        ],
        experience=[
            ExperienceRecord(
                role="Peer Learning Volunteer",
                organisation="Fictional university study program",
                description="Helped first-year students organise programming exercises.",
            )
        ],
        target_roles=["Data Analyst Intern"],
        is_confirmed=True,
    )
    flagged = _ungrounded_summary_terms(
        "She has experience as a Data Analyst Intern.",
        profile,
    )
    assert any("data analyst intern" in item.lower() for item in flagged)
    assert any("experience as" in item.lower() for item in flagged)

    ok = _ungrounded_summary_terms(
        "She is seeking a Data Analyst Intern role.",
        profile,
    )
    assert ok == []

    held_ok = _ungrounded_summary_terms(
        "She has experience as a Peer Learning Volunteer.",
        profile,
    )
    assert not any("peer learning volunteer" in item.lower() for item in held_ok)


def test_generate_cv_summary_falls_back_when_ungrounded(
    confirmed_profile: StudentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stub_call(_client, **_kwargs):
        return {
            "summary": "Led development of an enterprise Kubernetes platform."
        }

    monkeypatch.setattr(cvg, "call_gemini_json", stub_call)
    diagnostics: dict = {}
    text = generate_cv_summary(
        object(),
        confirmed_profile,
        "Python role",
        "AI Engineer",
        diagnostics=diagnostics,
    )
    assert diagnostics["path"] == "fallback"
    assert diagnostics["reason"] == "ungrounded"
    assert "Kubernetes" not in text
    assert "seeking" in text
