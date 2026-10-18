"""CV analyst tests. Deterministic: live Gemini is blocked here as well as in conftest."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import app.llm_client as llm_client
import streamlit_app as ui
from app.agent_engine import detect_thin_content
from app.cv_analyst import AnalysisIssue, _parse_items, _skills_items, analyse_cv
from app.cv_generator import (
    _ungrounded_summary_terms,
    generate_cv_markdown,
    profile_supports,
    ungrounded_terms,
)
from app.schemas import StudentProfile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_JOB = "graduate_ai_engineer"
RETAIL = (
    "Handled customer questions, worked with a small team during busy periods "
    "and followed written procedures."
)


def _rows(name: str) -> list[dict]:
    lines = (DATA_DIR / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


PROFILE_ROWS = {row["document_id"]: row for row in _rows("student_profiles.jsonl")}
JOBS = {row["document_id"]: row for row in _rows("jobs.jsonl")}
DEMO_IDS = sorted(PROFILE_ROWS)


@pytest.fixture(autouse=True)
def _no_live_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """The analyst calls Gemini through app.llm_client, so stub it at the source."""

    def no_call(*_args, **_kwargs):
        raise AssertionError("live Gemini call attempted in test_cv_analyst")

    monkeypatch.setattr(llm_client, "get_gemini_client", lambda: None)
    monkeypatch.setattr(llm_client, "call_gemini_json", no_call)


def _demo_profile(document_id: str) -> StudentProfile:
    profile = StudentProfile.model_validate(PROFILE_ROWS[document_id])
    profile.is_confirmed = True
    return profile


def _cv_sections(profile: StudentProfile, job_id: str = DEFAULT_JOB) -> dict[str, str]:
    """The exact sections the app hands the analyst: generated, then split."""
    job = JOBS[job_id]
    return ui._split_cv_markdown(generate_cv_markdown(profile, job, job["description"]))


def _vague(profile: StudentProfile, sections: dict[str, str]) -> list[AnalysisIssue]:
    issues = analyse_cv(profile, sections, JOBS[DEFAULT_JOB])
    return [issue for issue in issues if issue.type == "vague"]


def _maya_with_retail(text: str) -> tuple[StudentProfile, dict[str, str]]:
    """Maya's CV with the retail description replaced, as if edited in the editor."""
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    assert RETAIL in sections["Experience"]
    sections["Experience"] = sections["Experience"].replace(RETAIL, text)
    return maya, sections


# Public grounding wrappers in cv_generator


@pytest.mark.parametrize(
    "text",
    [
        "Built REST APIs in Python and Flask.",
        "Deployed services on Kubernetes and Terraform.",
        "Maya Chen has experience as a Software Engineering Intern.",
        "Led a team that shipped an enterprise platform.",
    ],
)
def test_ungrounded_terms_is_the_summary_check(text: str) -> None:
    maya = _demo_profile("student_profile_01")
    assert ungrounded_terms(text, maya, "AI Engineer") == _ungrounded_summary_terms(
        text, maya, job_title="AI Engineer"
    )


def test_ungrounded_terms_flags_invented_tools() -> None:
    maya = _demo_profile("student_profile_01")
    terms = ungrounded_terms("Deployed services on Kubernetes and Terraform.", maya)
    assert "Kubernetes" in terms
    assert "Terraform" in terms


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Flask", True),
        ("flask", True),
        ("pytest", True),
        ("REST APIs", True),
        ("Jupyter Notebook", True),
        ("Kubernetes", False),
        ("kubernetes", False),
        ("AWS", False),
        ("Go", False),
        ("C++", False),
        ("Lite", False),
        ("Hub", False),
    ],
)
def test_profile_supports_whole_phrases(phrase: str, expected: bool) -> None:
    assert profile_supports(phrase, _demo_profile("student_profile_01")) is expected


def test_profile_supports_counts_evidence_but_summary_grounding_is_unchanged() -> None:
    maya = _demo_profile("student_profile_01")
    claim = "Skilled in Python and Kubernetes."
    before = _ungrounded_summary_terms(claim, maya)
    assert "Kubernetes" in before
    assert not profile_supports("Kubernetes", maya)

    maya.skill_evidence.append(
        {"skill": "Kubernetes", "source": "kubernetes_certificate.png"}
    )

    assert profile_supports("Kubernetes", maya)
    assert profile_supports("kubernetes", maya)
    assert _ungrounded_summary_terms(claim, maya) == before


def test_profile_supports_empty_phrase() -> None:
    assert profile_supports("  ", StudentProfile())


@pytest.mark.parametrize("document_id", DEMO_IDS)
def test_unedited_list_items_are_supported(document_id: str) -> None:
    profile = _demo_profile(document_id)
    sections = _cv_sections(profile)
    listed = _skills_items(sections) + [
        tech for item in _parse_items(sections["Projects"]) for tech in item.technologies
    ]
    assert listed
    assert [item for item in listed if not profile_supports(item, profile)] == []


# Issue model and CV parsing


def test_analysis_issue_rejects_unknown_type_and_severity() -> None:
    fields = {
        "issue_id": "vague-00000000",
        "section": "Projects",
        "quote": "Built a thing.",
        "type": "vague",
        "severity": "high",
        "why": "w",
        "suggestion": "s",
        "question": "q?",
    }
    AnalysisIssue(**fields)
    with pytest.raises(ValidationError):
        AnalysisIssue(**{**fields, "type": "invented"})
    with pytest.raises(ValidationError):
        AnalysisIssue(**{**fields, "severity": "critical"})


def test_parse_items_reads_generated_entries() -> None:
    sections = _cv_sections(_demo_profile("student_profile_01"))
    projects = _parse_items(sections["Projects"])
    assert [item.title for item in projects] == [
        "Team Course Planner",
        "AI Coursework Prototype",
    ]
    assert projects[0].technologies == [
        "Python",
        "Flask",
        "SQLite",
        "REST APIs",
        "pytest",
        "Git",
    ]
    assert projects[0].description.startswith("Built a small web API")
    assert "Technologies" not in projects[0].description
    experience = _parse_items(sections["Experience"])
    assert [item.title for item in experience] == ["Customer Service Assistant"]
    assert experience[0].bullets == [RETAIL]


def test_parse_items_reads_edited_text_leniently() -> None:
    body = (
        "Loose line before any header\n"
        "**My Tool**\n"
        "Built it without a bullet\n"
        "-\n"
        "- Technologies: Rust"
    )
    items = _parse_items(body)
    assert [item.title for item in items] == ["", "My Tool"]
    assert items[0].bullets == ["Loose line before any header"]
    assert items[1].bullets == ["Built it without a bullet"]
    assert items[1].technologies == ["Rust"]


def test_skills_items_drop_category_labels() -> None:
    sections = {"Skills": "**Languages:** Python, Java\n- **Other Skills:** JSON"}
    assert _skills_items(sections) == ["Python", "Java", "JSON"]


# vague


EXPECTED_VAGUE = {
    "student_profile_01": ["Handled customer questions"],
    "student_profile_02": [],
    "student_profile_03": ["Helped first-year students"],
    "student_profile_04": ["Recorded support requests"],
    "student_profile_05": [],
}


@pytest.mark.parametrize("document_id", DEMO_IDS)
def test_demo_profiles_vague_issues(document_id: str) -> None:
    profile = _demo_profile(document_id)
    issues = _vague(profile, _cv_sections(profile))
    expected = EXPECTED_VAGUE[document_id]
    assert len(issues) == len(expected)
    for issue, prefix in zip(issues, expected, strict=True):
        assert issue.section == "Experience"
        assert issue.severity == "medium"
        assert issue.quote.startswith(prefix)


@pytest.mark.parametrize(
    "addition",
    [
        " using the SQL reporting screen",
        " for about 200 customers a day",
        " and reduced queue wait times",
    ],
    ids=["tool", "scale", "outcome"],
)
def test_any_one_signal_clears_vague(addition: str) -> None:
    maya, sections = _maya_with_retail(RETAIL.rstrip(".") + addition + ".")
    assert _vague(maya, sections) == []


def test_common_single_capital_tool_clears_vague() -> None:
    maya, sections = _maya_with_retail(
        "Used Excel to track stock and answered customer questions at the front counter."
    )
    assert _vague(maya, sections) == []


def test_profile_technology_outside_skills_clears_vague() -> None:
    maya, sections = _maya_with_retail(
        "Kept the Flask stock page up to date and answered customer questions "
        "at the front counter."
    )
    del sections["Skills"]
    assert _vague(maya, sections) == []


@pytest.mark.parametrize(
    "text",
    [
        "Handled IT questions, worked with a small team during busy periods "
        "and followed written procedures.",
        "Handled customer questions at UTS, worked with a small team during "
        "busy periods and followed written procedures.",
    ],
    ids=["stoplist", "institution"],
)
def test_non_tool_acronyms_keep_vague(text: str) -> None:
    maya, sections = _maya_with_retail(text)
    [issue] = _vague(maya, sections)
    assert issue.severity == "medium"


def test_organisation_initials_keep_vague() -> None:
    maya, sections = _maya_with_retail(
        "Handled customer questions for BRG, worked with a small team during "
        "busy periods and followed written procedures."
    )
    sections["Experience"] = sections["Experience"].replace(
        "Fictional retail store", "Big Retail Group"
    )
    [issue] = _vague(maya, sections)
    assert issue.severity == "medium"


def test_thin_description_is_high_severity() -> None:
    maya, sections = _maya_with_retail("Helped customers.")
    [issue] = _vague(maya, sections)
    assert issue.severity == "high"
    assert issue.quote == "Helped customers."
    assert "2 words" in issue.why
    assert "the Customer Service Assistant role" in issue.question


def test_entry_without_description_quotes_its_title() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections["Experience"] = (
        "**Customer Service Assistant, Fictional retail store, Casual employment**"
    )
    [issue] = _vague(maya, sections)
    assert issue.severity == "high"
    assert issue.quote == "Customer Service Assistant"
    assert "no description" in issue.why


def test_vague_high_covers_detect_thin_content() -> None:
    profile = _demo_profile("student_profile_02")
    profile.projects[0].description = "Built a retrieval assistant."
    thin = detect_thin_content(profile)
    assert [entry["title_or_role"] for entry in thin] == ["Coursework Retrieval Assistant"]

    high = [i for i in _vague(profile, _cv_sections(profile)) if i.severity == "high"]
    assert [issue.quote for issue in high] == ["Built a retrieval assistant."]


# Contract invariants


@pytest.mark.parametrize("job_id", sorted(JOBS))
@pytest.mark.parametrize("document_id", DEMO_IDS)
def test_every_quote_is_verbatim_from_its_section(document_id: str, job_id: str) -> None:
    profile = _demo_profile(document_id)
    sections = _cv_sections(profile, job_id)
    for issue in analyse_cv(profile, sections, JOBS[job_id]):
        assert issue.quote
        assert issue.quote in sections[issue.section]
        assert issue.why and issue.suggestion and issue.question


def test_issue_ids_are_stable_and_unique() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections["Projects"] += "\n**Extra**\n- Helped out.\n**Extra**\n- Helped out."
    first = [issue.issue_id for issue in analyse_cv(maya, sections, None)]
    second = [issue.issue_id for issue in analyse_cv(maya, sections, None)]
    assert first == second
    assert len(set(first)) == len(first) == 3
    assert all(re.fullmatch(r"vague-[0-9a-f]{8}(-\d+)?", issue_id) for issue_id in first)


def test_issues_are_sorted_most_severe_first() -> None:
    maya, sections = _maya_with_retail("Helped customers.")
    sections["Projects"] += (
        "\n**Side project**\n"
        "- Organised weekly study sessions for classmates in the library."
    )
    issues = analyse_cv(maya, sections, None)
    assert [(issue.severity, issue.section) for issue in issues] == [
        ("high", "Experience"),
        ("medium", "Projects"),
    ]
