"""CV analyst tests. Deterministic: live Gemini is blocked here as well as in conftest."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import app.cv_analyst as cv_analyst
import app.llm_client as llm_client
import streamlit_app as ui
from app.agent_engine import detect_thin_content
from app.cv_analyst import (
    _BANNED_WORDING_RE,
    _FEEDBACK_RESPONSE_SCHEMA,
    JOB_AD_SECTION,
    MAX_REQUIREMENT_ISSUES,
    SEVERITY_ORDER,
    UNAVAILABLE_RETRY_SECONDS,
    AnalysisIssue,
    _parse_items,
    _sections_from_markdown,
    _skills_items,
    add_skill_evidence,
    analyse_cv,
    main,
)
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
# Rule 3 of the brief: no claim that the student lacks something, and no
# prediction about hiring.
BANNED_WORDING = re.compile(
    r"\b(?:lack\w*|don't have|do not have|not qualified|unqualified|hire\w*|"
    r"hiring|shortlist\w*|reject\w*|your chances|not a fit)\b",
    re.IGNORECASE,
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
        if issue.section == JOB_AD_SECTION:
            assert issue.quote in JOBS[job_id]["description"]
        else:
            assert issue.quote in sections[issue.section]
        assert issue.why and issue.suggestion and issue.question


def test_issue_ids_are_stable_and_unique() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections["Projects"] += "\n**Extra**\n- Helped out.\n**Extra**\n- Helped out."
    first = [issue.issue_id for issue in analyse_cv(maya, sections, None)]
    second = [issue.issue_id for issue in analyse_cv(maya, sections, None)]
    assert first == second
    assert len(set(first)) == len(first)
    id_re = r"(vague|no_evidence|ungrounded_edit|low_relevance)-[0-9a-f]{8}(-\d+)?"
    assert all(re.fullmatch(id_re, issue_id) for issue_id in first)
    assert sum(issue_id.endswith("-2") for issue_id in first) == 1


def test_issues_are_sorted_most_severe_first() -> None:
    maya, sections = _maya_with_retail("Helped customers.")
    sections["Projects"] += (
        "\n**Side project**\n"
        "- Organised weekly study sessions for classmates in the library."
    )
    issues = analyse_cv(maya, sections, None)
    ranks = [SEVERITY_ORDER.index(issue.severity) for issue in issues]
    assert ranks == sorted(ranks)
    assert [(i.severity, i.section) for i in issues if i.type == "vague"] == [
        ("high", "Experience"),
        ("medium", "Projects"),
    ]


@pytest.mark.parametrize("job_id", sorted(JOBS))
@pytest.mark.parametrize("document_id", DEMO_IDS)
def test_every_demo_profile_gets_section_issues(document_id: str, job_id: str) -> None:
    profile = _demo_profile(document_id)
    sections = _cv_sections(profile, job_id)
    issues = analyse_cv(profile, sections, JOBS[job_id])
    assert issues
    for issue in issues:
        assert issue.section in sections or issue.section == JOB_AD_SECTION
        assert issue.why


def test_templates_never_claim_the_student_lacks_anything() -> None:
    for document_id in DEMO_IDS:
        profile = _demo_profile(document_id)
        for issue in analyse_cv(profile, _cv_sections(profile), JOBS[DEFAULT_JOB]):
            text = " ".join((issue.why, issue.suggestion, issue.question))
            assert not BANNED_WORDING.search(text), text
            # Templates are the fallback, so they must pass the Gemini check too.
            assert not _BANNED_WORDING_RE.search(text), text
            if issue.type == "no_evidence":
                assert "not shown in your CV" in issue.why


# ungrounded_edit


@pytest.mark.parametrize("job_id", sorted(JOBS))
@pytest.mark.parametrize("document_id", DEMO_IDS)
def test_unedited_cvs_have_no_ungrounded_edit(document_id: str, job_id: str) -> None:
    profile = _demo_profile(document_id)
    sections = _cv_sections(profile, job_id)
    assert "cv_layout" in sections["Summary"]
    issues = analyse_cv(profile, sections, JOBS[job_id])
    assert [i for i in issues if i.type == "ungrounded_edit"] == []
    assert not any("cv_layout" in i.quote or "cv_layout" in i.why for i in issues)


@pytest.mark.parametrize(
    ("section", "edit", "quotes", "named"),
    [
        ("Skills", lambda s: s + ", Kubernetes", ["Kubernetes"], ["Kubernetes"]),
        ("Skills", lambda s: s + "\nkubernetes", ["kubernetes"], ["kubernetes"]),
        ("Skills", lambda s: s + "\nAWS, Go, C++", ["AWS", "Go", "C++"], ["AWS"]),
        (
            "Projects",
            lambda s: s.replace("pytest, Git", "pytest, Git, Terraform"),
            ["Terraform"],
            ["Terraform"],
        ),
        (
            "Experience",
            lambda s: s + "\n- Led a team of 10 engineers using TensorFlow.",
            ["Led a team of 10 engineers using TensorFlow."],
            ["TensorFlow", "led a team"],
        ),
        (
            "Professional Summary",
            lambda s: s + " Worked as a Software Engineering Intern.",
            ["Worked as a Software Engineering Intern."],
            ["experience as Software Engineering Intern"],
        ),
        (
            "Summary",
            lambda s: s.replace("# Maya Chen", "# Maya Chen, Senior Engineer at Google"),
            ["Maya Chen, Senior Engineer at Google"],
            ["Google"],
        ),
    ],
    ids=[
        "skill-appended",
        "skill-lowercase",
        "short-skills",
        "technology",
        "experience-sentence",
        "role-framing",
        "header",
    ],
)
def test_typed_unsupported_text_is_ungrounded_edit(section, edit, quotes, named) -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections[section] = edit(sections[section])
    issues = analyse_cv(maya, sections, JOBS[DEFAULT_JOB])
    flagged = [issue for issue in issues if issue.type == "ungrounded_edit"]
    assert [issue.quote for issue in flagged] == quotes
    assert all(issue.section == section and issue.severity == "high" for issue in flagged)
    assert all(issue.quote in sections[section] for issue in flagged)
    for term in named:
        assert term in flagged[0].why


def test_ungrounded_skill_is_not_also_no_evidence() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections["Skills"] += ", Kubernetes"
    about = [i.type for i in analyse_cv(maya, sections, None) if "Kubernetes" in i.quote]
    assert about == ["ungrounded_edit"]


def test_layout_comment_is_never_analysed() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    baseline = analyse_cv(maya, sections, JOBS[DEFAULT_JOB])
    comment = "<!-- cv_layout: single -->"
    for name in ("Skills", "Projects", "Experience", "Professional Summary"):
        sections[name] = f"{comment}\n{sections[name]}\n{comment}"
    issues = analyse_cv(maya, sections, JOBS[DEFAULT_JOB])
    assert [i.issue_id for i in issues] == [i.issue_id for i in baseline]
    assert not any("cv_layout" in i.quote for i in issues)


# The demo loop: flag, add evidence, re-analyse, cleared


@pytest.mark.parametrize(
    ("section", "edit"),
    [
        ("Skills", lambda s: s + ", Kubernetes"),
        ("Experience", lambda s: s + "\n- Deployed the stock app on Kubernetes."),
    ],
    ids=["skills-item", "experience-sentence"],
)
def test_evidence_clears_a_flagged_edit(section, edit) -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections[section] = edit(sections[section])

    issues = analyse_cv(maya, sections, JOBS[DEFAULT_JOB])
    before = [issue for issue in issues if "Kubernetes" in issue.quote]
    assert [(i.type, i.severity) for i in before] == [("ungrounded_edit", "high")]

    result = add_skill_evidence(maya, "Kubernetes", "kubernetes_certificate.png")
    assert result["status"] == "success"

    after = analyse_cv(maya, sections, JOBS[DEFAULT_JOB])
    assert not any("Kubernetes" in i.quote or "Kubernetes" in i.why for i in after)


# add_skill_evidence


def test_add_skill_evidence_keeps_existing_entries() -> None:
    maya = _demo_profile("student_profile_01")
    original = [dict(entry) for entry in maya.skill_evidence]
    maya.skill_evidence[0]["note"] = "extra key from an older writer"

    result = add_skill_evidence(maya, "  Flask ", " flask_screenshot.png ")

    assert result == {"status": "success", "updated_fields": ["skill_evidence"]}
    assert maya.skill_evidence[:-1] == original
    assert maya.skill_evidence[-1] == {"skill": "Flask", "source": "flask_screenshot.png"}


def test_add_skill_evidence_skips_exact_duplicates() -> None:
    maya = _demo_profile("student_profile_01")
    existing = dict(maya.skill_evidence[0])
    count = len(maya.skill_evidence)
    assert add_skill_evidence(maya, existing["skill"], existing["source"]) == {
        "status": "skipped",
        "reason": "duplicate",
    }
    assert add_skill_evidence(maya, "Flask", "a.png")["status"] == "success"
    assert add_skill_evidence(maya, "Flask", "a.png")["status"] == "skipped"
    assert len(maya.skill_evidence) == count + 1


@pytest.mark.parametrize(
    ("skill", "source"), [("", "a.png"), ("Flask", "   "), (None, "a.png")]
)
def test_add_skill_evidence_rejects_empty_values(skill, source) -> None:
    maya = _demo_profile("student_profile_01")
    before = [dict(entry) for entry in maya.skill_evidence]
    result = add_skill_evidence(maya, skill, source)
    assert result["status"] == "error"
    assert maya.skill_evidence == before


# no_evidence


def test_maya_no_evidence_is_the_undescribed_technologies() -> None:
    maya = _demo_profile("student_profile_01")
    issues = analyse_cv(maya, _cv_sections(maya), None)
    issues = [issue for issue in issues if issue.type == "no_evidence"]
    assert {i.quote for i in issues} == {"Flask", "SQLite", "prompt design", "JSON"}
    assert all(i.section == "Projects" and i.severity == "low" for i in issues)
    flask = next(i for i in issues if i.quote == "Flask")
    assert "the Team Course Planner project" in flask.why


def test_tool_calling_clears_against_tool_call_agent() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    assert "tool-call agent" in sections["Projects"]
    quotes = [i.quote for i in analyse_cv(maya, sections, None) if i.type == "no_evidence"]
    assert "tool calling" not in quotes

    sections["Projects"] = sections["Projects"].replace("tool-call agent", "agent")
    quotes = [i.quote for i in analyse_cv(maya, sections, None) if i.type == "no_evidence"]
    assert "tool calling" in quotes


def test_evidence_entry_clears_no_evidence() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    add_skill_evidence(maya, "Flask", "flask_screenshot.png")
    quotes = [i.quote for i in analyse_cv(maya, sections, None) if i.type == "no_evidence"]
    assert "Flask" not in quotes
    assert "SQLite" in quotes


def test_skills_only_claim_is_medium() -> None:
    maya = _demo_profile("student_profile_01")
    maya.technical_skills.append("Docker basics")
    issues = analyse_cv(maya, _cv_sections(maya), None)
    issues = [issue for issue in issues if issue.quote == "Docker basics"]
    assert [(i.type, i.section, i.severity) for i in issues] == [
        ("no_evidence", "Skills", "medium")
    ]


def test_achievement_needs_evidence_until_added() -> None:
    maya = _demo_profile("student_profile_01")
    maya.activities.append("Won first prize at the UTS Hackathon")
    sections = _cv_sections(maya)
    claim = "Won first prize at the UTS Hackathon"
    issues = [i for i in analyse_cv(maya, sections, None) if i.quote == claim]
    assert [(i.type, i.section, i.severity) for i in issues] == [
        ("no_evidence", "Activities", "low")
    ]

    add_skill_evidence(maya, "first prize at the UTS Hackathon", "hackathon_certificate.png")
    assert not any(i.quote == claim for i in analyse_cv(maya, sections, None))


def test_typed_achievement_is_ungrounded_not_no_evidence() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections["Activities"] += "\n- Won the Atlassian Hackathon award"
    about = [i.type for i in analyse_cv(maya, sections, None) if "Atlassian" in i.quote]
    assert about == ["ungrounded_edit"]


def test_typed_achievement_is_named_as_an_achievement() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    claim = "Won Employee of the Month for handling customer questions during busy periods."
    sections["Experience"] += f"\n- {claim}"

    [issue] = [i for i in analyse_cv(maya, sections, None) if i.quote == claim]
    assert (issue.type, issue.severity) == ("ungrounded_edit", "high")
    assert issue.why.startswith("This achievement isn't in your confirmed profile")
    assert "Employee" not in issue.why and "Month" not in issue.why
    assert "Employee" not in issue.question

    add_skill_evidence(maya, "Employee of the Month", "employee_of_the_month.png")
    assert not any(i.quote == claim for i in analyse_cv(maya, sections, None))


# low_relevance


def _job(description: str, **extra) -> dict:
    return {
        "document_id": "custom_job",
        "title": "Backend Intern",
        "description": description,
        **extra,
    }


def _job_ad_issues(
    profile: StudentProfile, sections: dict[str, str], job: dict
) -> list[AnalysisIssue]:
    return [i for i in analyse_cv(profile, sections, job) if i.section == JOB_AD_SECTION]


def _entry_relevance_issues(
    profile: StudentProfile, sections: dict[str, str], job: dict | None
) -> list[AnalysisIssue]:
    issues = analyse_cv(profile, sections, job)
    return [
        i for i in issues if i.type == "low_relevance" and i.section != JOB_AD_SECTION
    ]


def test_unrelated_entry_is_low_relevance() -> None:
    maya = _demo_profile("student_profile_01")
    issues = _entry_relevance_issues(maya, _cv_sections(maya), JOBS[DEFAULT_JOB])
    assert [(i.section, i.quote, i.severity) for i in issues] == [
        ("Experience", RETAIL, "low")
    ]
    assert "your CV lists (Python)" in issues[0].why
    assert issues[0].suggestion == (
        "If this work used Python, name it and say what you did with it. "
        "If it didn't, you don't need to change this entry."
    )
    assert issues[0].question == "Did the Customer Service Assistant role involve Python?"


@pytest.mark.parametrize("job_id", sorted(JOBS))
def test_relevant_but_unnamed_work_is_not_told_to_shrink(job_id: str) -> None:
    ethan = _demo_profile("student_profile_05")
    for issue in _entry_relevance_issues(ethan, _cv_sections(ethan, job_id), JOBS[job_id]):
        assert "short" not in issue.suggestion and "lower" not in issue.suggestion
        assert "you don't need to change this entry" in issue.suggestion


def test_no_entry_is_flagged_when_the_job_names_no_cv_skill() -> None:
    maya = _demo_profile("student_profile_01")
    job = _job("Requirements: Kafka.")
    assert _entry_relevance_issues(maya, _cv_sections(maya), job) == []


def test_unsupported_skill_does_not_count_as_a_job_skill() -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections["Skills"] += ", Kubernetes"
    job = _job("Requirements: Python, Kubernetes.")
    issues = _entry_relevance_issues(maya, sections, job)
    assert [i.quote for i in issues] == [RETAIL]
    assert "Kubernetes" not in issues[0].why


def test_no_job_means_no_low_relevance() -> None:
    maya = _demo_profile("student_profile_01")
    issues = analyse_cv(maya, _cv_sections(maya), None)
    assert not any(issue.type == "low_relevance" for issue in issues)


@pytest.mark.parametrize(
    ("description", "quote"),
    [
        ("Requirements: Experience with Docker.", "Docker"),
        (
            "Requirements: Docker experience, working knowledge and strong skills.",
            "Docker experience",
        ),
    ],
    ids=["lead-in", "generic-words"],
)
def test_docker_requirement_is_flagged_for_a_cv_without_docker(
    description: str, quote: str
) -> None:
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    # The generic words are now in the CV, so only Docker can address the item.
    sections["Professional Summary"] += (
        " Strong working knowledge and good basic experience and skills."
    )
    job = _job(description)
    [issue] = _job_ad_issues(maya, sections, job)
    assert (issue.type, issue.quote, issue.severity) == ("low_relevance", quote, "medium")
    assert issue.quote in job["description"]
    assert issue.why == f'The job ad asks for "{quote}", and your CV doesn\'t mention it.'
    assert f'"{quote}"' in issue.question


def test_docker_requirement_is_addressed_by_a_cv_with_docker() -> None:
    noah = _demo_profile("student_profile_04")
    job = _job("Requirements: Experience with Docker.")
    assert _job_ad_issues(noah, _cv_sections(noah), job) == []


def test_preferred_requirement_is_low() -> None:
    maya = _demo_profile("student_profile_01")
    issues = _job_ad_issues(
        maya, _cv_sections(maya), _job("Requirements: Python. Nice to have: Kubernetes.")
    )
    assert [(i.quote, i.severity) for i in issues] == [("Kubernetes", "low")]
    assert 'lists "Kubernetes" as preferred' in issues[0].why


def test_filler_requirement_items_are_dropped() -> None:
    maya = _demo_profile("student_profile_01")
    job = _job(
        "Requirements: Kafka, or similar frameworks; other cloud platforms; "
        "related tools; equivalent experience; comparable systems."
    )
    assert [i.quote for i in _job_ad_issues(maya, _cv_sections(maya), job)] == ["Kafka"]


def test_computer_vision_job_has_no_similar_frameworks_issue() -> None:
    maya = _demo_profile("student_profile_01")
    job = JOBS["computer_vision_engineer"]
    quotes = [i.quote for i in _job_ad_issues(maya, _cv_sections(maya), job)]
    assert quotes
    assert not any(quote.lower().startswith("similar") for quote in quotes)


def test_desirable_block_without_colon_is_preferred() -> None:
    maya = _demo_profile("student_profile_01")
    job = _job(
        "Requirements: Python; familiarity with CI/CD. "
        "Desirable experience includes Kubernetes, Terraform."
    )
    issues = _job_ad_issues(maya, _cv_sections(maya), job)
    assert [(i.quote, i.severity) for i in issues] == [
        ("CI/CD", "medium"),
        ("Kubernetes", "low"),
        ("Terraform", "low"),
    ]


def test_eligibility_clauses_are_skipped() -> None:
    maya = _demo_profile("student_profile_01")
    # Words Maya's CV lacks, so keeping these clauses would flag them.
    job = _job(
        "Requirements: A master's degree in astrophysics; a PhD or postgraduate "
        "study in oceanography; Kafka."
    )
    assert [i.quote for i in _job_ad_issues(maya, _cv_sections(maya), job)] == ["Kafka"]


@pytest.mark.parametrize("job_id", sorted(JOBS))
def test_demo_jobs_never_flag_eligibility(job_id: str) -> None:
    eligibility = re.compile(r"degree|bachelor|master|postgraduate|years?\b", re.I)
    maya = _demo_profile("student_profile_01")
    for issue in _job_ad_issues(maya, _cv_sections(maya, job_id), JOBS[job_id]):
        assert not eligibility.search(issue.quote), issue.quote


def test_requirement_issues_are_capped_required_first() -> None:
    maya = _demo_profile("student_profile_01")
    required = ["Rust", "Kafka", "Spark", "Hadoop", "Scala", "Erlang", "Haskell"]
    job = _job(f"Nice to have: Elixir. Requirements: {', '.join(required)}.")
    issues = _job_ad_issues(maya, _cv_sections(maya), job)
    assert [i.quote for i in issues] == required[:MAX_REQUIREMENT_ISSUES]
    assert all(issue.severity == "medium" for issue in issues)


def test_requirements_list_from_the_job_ad_parser_is_used() -> None:
    maya = _demo_profile("student_profile_01")
    job = _job("", requirements=["Experience with Terraform", "Python"])
    issues = _job_ad_issues(maya, _cv_sections(maya), job)
    assert [(i.quote, i.severity) for i in issues] == [("Terraform", "medium")]


# Standalone run: python -m app.cv_analyst <profile_id> <job_id>


@pytest.mark.parametrize("document_id", DEMO_IDS)
def test_standalone_split_matches_the_app(document_id: str) -> None:
    job = JOBS[DEFAULT_JOB]
    markdown = generate_cv_markdown(_demo_profile(document_id), job, job["description"])
    assert _sections_from_markdown(markdown) == ui._split_cv_markdown(markdown)


def test_standalone_run_prints_issues(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["student_profile_01", DEFAULT_JOB]) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "Maya Chen (student_profile_01) vs Graduate AI Engineer (graduate_ai_engineer)"
    )
    assert "[medium] vague | Experience" in out
    assert "low_relevance | Job ad" in out


def test_standalone_run_json_matches_analyse_cv(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["student_profile_01", DEFAULT_JOB, "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    maya = _demo_profile("student_profile_01")
    issues = analyse_cv(maya, _cv_sections(maya), JOBS[DEFAULT_JOB])
    assert printed == [issue.model_dump() for issue in issues]


@pytest.mark.parametrize(
    "argv", [["nobody", DEFAULT_JOB], ["student_profile_01", "no_such_job"]]
)
def test_standalone_run_rejects_unknown_ids(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


# Gemini feedback, with stubs only


def _echo_feedback(payload: dict) -> dict:
    """A well-behaved response: every issue, its quote copied, plain text."""
    return {
        "issues": [
            {
                "issue_id": item["issue_id"],
                "quote": item["quote"],
                "why": "llm-written: a reader cannot see what this part shows.",
                "suggestion": "llm-written: add one detail, such as [number of users].",
                "question": "llm-written: what changed because of this work?",
            }
            for item in payload["issues"]
        ]
    }


class _GeminiStub:
    """Stands in for app.llm_client: records each call, answers from the payload."""

    def __init__(
        self, monkeypatch, respond=_echo_feedback, error=None, errors=None
    ) -> None:
        """error is raised on every call; errors are raised one per call, first."""
        self.calls: list[dict] = []
        self.respond = respond
        self.error = error
        self.errors = list(errors or [])
        monkeypatch.setattr(llm_client, "get_gemini_client", lambda: "stub-client")
        monkeypatch.setattr(llm_client, "call_gemini_json", self)

    def __call__(self, client, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        if self.error is not None:
            raise self.error
        return self.respond(json.loads(kwargs["user_content"]))


class _SdkError(Exception):
    """Stands in for google.genai.errors.ServerError, which carries .code."""

    def __init__(self, code: int) -> None:
        super().__init__(str(code))
        self.code = code


def _api_error(code: int, status: str = "UNAVAILABLE") -> llm_client.GeminiAPIError:
    """What llm_client raises: its own error, with the SDK error as the cause."""
    error = llm_client.GeminiAPIError(f"Gemini generate_content failed: {code} {status}.")
    error.__cause__ = _SdkError(code)
    return error


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry waits instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(cv_analyst.time, "sleep", recorded.append)
    return recorded


def _maya_run(job: dict | None = None, sections: dict[str, str] | None = None):
    """Maya's issues with Gemini on, their template twins, and the diagnostics."""
    maya = _demo_profile("student_profile_01")
    sections = sections or _cv_sections(maya)
    job = job or JOBS[DEFAULT_JOB]
    template = analyse_cv(maya, sections, job, use_gemini=False)
    diagnostics: dict = {}
    issues = analyse_cv(maya, sections, job, diagnostics=diagnostics)
    return issues, template, diagnostics


def _first_issue_changed(field_name: str, value: str):
    def respond(payload: dict) -> dict:
        response = _echo_feedback(payload)
        response["issues"][0][field_name] = value
        return response

    return respond


def test_valid_gemini_text_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _GeminiStub(monkeypatch)
    issues, template, diagnostics = _maya_run()
    assert len(stub.calls) == 1
    assert issues and all(issue.why.startswith("llm-written") for issue in issues)
    assert diagnostics["path"] == "gemini"
    assert all(
        entry == {"source": "gemini", "reason": "", "detail": ""}
        for entry in diagnostics["issues"].values()
    )
    call = stub.calls[0]
    assert call["response_schema"] == _FEEDBACK_RESPONSE_SCHEMA
    assert "model" not in call  # the model name stays in llm_client


def test_gemini_cannot_change_type_severity_or_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(payload: dict) -> dict:
        response = _echo_feedback(payload)
        for item in response["issues"]:
            item.update(type="vague", severity="low", section="Skills")
        return response

    _GeminiStub(monkeypatch, respond)
    issues, template, _ = _maya_run()
    fixed = ("issue_id", "type", "severity", "section", "quote")
    assert [[getattr(i, f) for f in fixed] for i in issues] == [
        [getattr(t, f) for f in fixed] for t in template
    ]


def test_payload_redacts_the_name_and_skips_private_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _GeminiStub(monkeypatch)
    maya = _demo_profile("student_profile_01")
    sections = _cv_sections(maya)
    sections["Summary"] = sections["Summary"].replace(
        "# Maya Chen", "# Maya Chen, Senior Engineer at Google"
    )
    sections["Professional Summary"] += " Worked as a Software Engineering Intern."
    issues, template, diagnostics = _maya_run(sections=sections)

    sent = stub.calls[0]["user_content"]
    for private in ("Maya", "Chen", "Google", "github.com", "linkedin"):
        assert private not in sent
    assert "[name]" in sent
    [header] = [i for i in issues if i.section == "Summary"]
    assert diagnostics["issues"][header.issue_id]["reason"] == "private_section"
    assert header.why == next(t.why for t in template if t.issue_id == header.issue_id)


def test_ungrounded_gemini_field_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _GeminiStub(
        monkeypatch,
        _first_issue_changed("suggestion", "Mention the Kubernetes cluster you ran."),
    )
    issues, template, diagnostics = _maya_run()
    assert issues[0].model_dump() == template[0].model_dump()
    entry = diagnostics["issues"][issues[0].issue_id]
    assert entry["source"] == "template" and entry["reason"] == "ungrounded"
    assert "Kubernetes" in entry["detail"]
    assert all(issue.why.startswith("llm-written") for issue in issues[1:])
    assert diagnostics["path"] == "partial"


def test_job_ad_terms_from_the_quote_are_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(payload: dict) -> dict:
        response = _echo_feedback(payload)
        for item in response["issues"]:
            if item["quote"] == "TensorFlow":
                item["why"] = "The ad asks for TensorFlow, which your CV does not show."
        return response

    _GeminiStub(monkeypatch, respond)
    issues, _, diagnostics = _maya_run(job=_job("Requirements: TensorFlow."))
    [issue] = [i for i in issues if i.section == JOB_AD_SECTION]
    assert issue.why == "The ad asks for TensorFlow, which your CV does not show."
    assert diagnostics["issues"][issue.issue_id]["source"] == "gemini"


@pytest.mark.parametrize(
    "phrase",
    [
        "you lack Python",
        "you don't have the skills",
        "you are not qualified",
        "they will not hire you",
        "an employer will skip it",
    ],
)
def test_banned_wording_falls_back(monkeypatch: pytest.MonkeyPatch, phrase: str) -> None:
    _GeminiStub(monkeypatch, _first_issue_changed("why", f"llm-written: {phrase}."))
    issues, template, diagnostics = _maya_run()
    assert issues[0].model_dump() == template[0].model_dump()
    assert diagnostics["issues"][issues[0].issue_id]["reason"] == "banned"


def test_text_written_for_another_quote_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(payload: dict) -> dict:
        response = _echo_feedback(payload)
        response["issues"][0]["quote"] = response["issues"][1]["quote"]
        return response

    _GeminiStub(monkeypatch, respond)
    issues, template, diagnostics = _maya_run()
    assert issues[0].model_dump() == template[0].model_dump()
    assert diagnostics["issues"][issues[0].issue_id]["reason"] == "quote_mismatch"
    assert issues[1].why.startswith("llm-written")


@pytest.mark.parametrize(
    ("field_name", "value", "reason"),
    [("why", "   ", "empty_field"), ("question", "x" * 301, "too_long")],
)
def test_empty_or_long_field_falls_back(
    monkeypatch: pytest.MonkeyPatch, field_name: str, value: str, reason: str
) -> None:
    _GeminiStub(monkeypatch, _first_issue_changed(field_name, value))
    issues, template, diagnostics = _maya_run()
    assert issues[0].model_dump() == template[0].model_dump()
    assert diagnostics["issues"][issues[0].issue_id]["reason"] == reason


def test_missing_issue_falls_back_and_unknown_ids_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(payload: dict) -> dict:
        response = _echo_feedback(payload)
        dropped = response["issues"].pop(0)
        response["issues"].append({**dropped, "issue_id": "vague-ffffffff"})
        return response

    _GeminiStub(monkeypatch, respond)
    issues, template, diagnostics = _maya_run()
    assert issues[0].model_dump() == template[0].model_dump()
    assert diagnostics["issues"][issues[0].issue_id]["reason"] == "missing"
    assert "vague-ffffffff" not in diagnostics["issues"]
    assert [i.issue_id for i in issues] == [t.issue_id for t in template]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (llm_client.GeminiAPIError("quota exceeded"), "api"),
        (llm_client.GeminiJSONParseError("not JSON"), "parse"),
    ],
)
def test_gemini_errors_fall_back_to_templates(
    monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str
) -> None:
    _GeminiStub(monkeypatch, error=error)
    issues, template, diagnostics = _maya_run()
    assert [i.model_dump() for i in issues] == [t.model_dump() for t in template]
    assert (diagnostics["path"], diagnostics["reason"]) == ("template", reason)
    assert str(error) in diagnostics["detail"]
    assert {entry["reason"] for entry in diagnostics["issues"].values()} == {reason}


def test_response_without_an_issues_list_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _GeminiStub(monkeypatch, lambda payload: {"feedback": []})
    issues, template, diagnostics = _maya_run()
    assert [i.model_dump() for i in issues] == [t.model_dump() for t in template]
    assert diagnostics["reason"] == "empty"


def test_no_client_means_templates_and_no_call() -> None:
    issues, template, diagnostics = _maya_run()
    assert [i.model_dump() for i in issues] == [t.model_dump() for t in template]
    assert (diagnostics["path"], diagnostics["reason"]) == ("template", "no_client")


def test_use_gemini_false_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _GeminiStub(monkeypatch)
    maya = _demo_profile("student_profile_01")
    diagnostics: dict = {}
    analyse_cv(maya, _cv_sections(maya), None, use_gemini=False, diagnostics=diagnostics)
    assert stub.calls == []
    assert diagnostics["reason"] == "disabled"


def test_no_issues_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _GeminiStub(monkeypatch)
    diagnostics: dict = {}
    maya = _demo_profile("student_profile_01")
    assert analyse_cv(maya, {}, None, diagnostics=diagnostics) == []
    assert stub.calls == []
    assert diagnostics["reason"] == "no_issues"


def test_standalone_run_stays_deterministic_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = _GeminiStub(monkeypatch)
    assert main(["student_profile_01", DEFAULT_JOB]) == 0
    assert stub.calls == []
    out = capsys.readouterr().out
    assert re.search(r"Text: template for all \d+ issues \(Gemini not requested", out)


def test_standalone_run_gemini_flag_opts_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = _GeminiStub(monkeypatch)
    assert main(["student_profile_01", DEFAULT_JOB, "--gemini"]) == 0
    assert len(stub.calls) == 1
    out = capsys.readouterr().out
    assert re.search(r"Text: Gemini for (\d+) of \1 issues", out)
    assert "llm-written" in out
    assert "   text:       gemini" in out


@pytest.mark.parametrize("placeholder", ["[name]", "[Name]"])
def test_name_placeholder_becomes_the_first_name(
    monkeypatch: pytest.MonkeyPatch, placeholder: str
) -> None:
    _GeminiStub(
        monkeypatch,
        _first_issue_changed("why", f"{placeholder}, this line is easy to miss."),
    )
    issues, _, diagnostics = _maya_run()
    assert issues[0].why == "Maya, this line is easy to miss."
    assert diagnostics["issues"][issues[0].issue_id]["source"] == "gemini"
    shown = " ".join(f"{i.why} {i.suggestion} {i.question}" for i in issues)
    assert "[name]" not in shown.lower()


def test_name_placeholder_without_a_name_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _GeminiStub(
        monkeypatch, _first_issue_changed("why", "[name], this line is easy to miss.")
    )
    nameless = _demo_profile("student_profile_01")
    nameless.name = ""
    sections = _cv_sections(nameless)
    template = analyse_cv(nameless, sections, JOBS[DEFAULT_JOB], use_gemini=False)
    diagnostics: dict = {}
    issues = analyse_cv(nameless, sections, JOBS[DEFAULT_JOB], diagnostics=diagnostics)
    assert issues[0].model_dump() == template[0].model_dump()
    assert diagnostics["issues"][issues[0].issue_id]["reason"] == "name_placeholder"


def test_invented_number_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    invented = "Say you served 50 customers a shift."
    _GeminiStub(monkeypatch, _first_issue_changed("suggestion", invented))
    issues, template, diagnostics = _maya_run()
    assert issues[0].model_dump() == template[0].model_dump()
    entry = diagnostics["issues"][issues[0].issue_id]
    assert (entry["reason"], entry["detail"]) == ("invented_number", "suggestion: 50")


def test_numbers_from_the_issue_and_placeholders_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "llm-written: add a Flask line from the 43007 work, such as [e.g. 3 endpoints]."

    def respond(payload: dict) -> dict:
        response = _echo_feedback(payload)
        for item in response["issues"]:
            if item["quote"] == "Flask":
                item["suggestion"] = text
        return response

    _GeminiStub(monkeypatch, respond)
    issues, _, diagnostics = _maya_run()
    [flask] = [i for i in issues if i.quote == "Flask"]
    assert "43007" in _cv_sections(_demo_profile("student_profile_01"))["Projects"]
    assert flask.suggestion == text
    assert diagnostics["issues"][flask.issue_id]["source"] == "gemini"


def test_busy_model_is_retried_once_then_used(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    stub = _GeminiStub(monkeypatch, errors=[_api_error(503)])
    issues, _, diagnostics = _maya_run()
    assert len(stub.calls) == 2
    assert sleeps == [UNAVAILABLE_RETRY_SECONDS] == [2]
    assert (diagnostics["path"], diagnostics["retries"]) == ("gemini", 1)
    assert all(issue.why.startswith("llm-written") for issue in issues)


def test_busy_model_twice_falls_back(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    stub = _GeminiStub(monkeypatch, errors=[_api_error(503), _api_error(503)])
    issues, template, diagnostics = _maya_run()
    assert len(stub.calls) == 2
    assert sleeps == [2]
    assert [i.model_dump() for i in issues] == [t.model_dump() for t in template]
    assert (diagnostics["path"], diagnostics["reason"], diagnostics["retries"]) == (
        "template",
        "api",
        1,
    )
    assert "503 UNAVAILABLE" in diagnostics["detail"]


@pytest.mark.parametrize(
    "error",
    [
        _api_error(429, "RESOURCE_EXHAUSTED"),
        llm_client.GeminiAPIError("quota exceeded"),
        llm_client.GeminiJSONParseError("not JSON"),
    ],
    ids=["api-429", "api-no-cause", "parse"],
)
def test_only_a_busy_model_is_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], error: Exception
) -> None:
    stub = _GeminiStub(monkeypatch, errors=[error])
    _, _, diagnostics = _maya_run()
    assert len(stub.calls) == 1
    assert sleeps == []
    assert diagnostics["retries"] == 0
    assert diagnostics["path"] == "template"


def test_report_prints_a_batch_error_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sleeps: list[float],
) -> None:
    _GeminiStub(monkeypatch, error=_api_error(503))
    assert main(["student_profile_01", DEFAULT_JOB, "--gemini"]) == 0
    out = capsys.readouterr().out
    assert out.count("generate_content failed") == 1
    [text_line] = [line for line in out.splitlines() if line.startswith("Text: ")]
    assert "retried once after 503 UNAVAILABLE" in text_line
    assert "batch error: Gemini generate_content failed: 503 UNAVAILABLE." in text_line
    per_issue = [line for line in out.splitlines() if line.startswith("   text:")]
    assert per_issue and set(per_issue) == {"   text:       template (api)"}
