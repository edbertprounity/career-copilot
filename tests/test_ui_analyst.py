"""CV analyst panel tests with AppTest. Gemini is stubbed; no live calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.cv_analyst as cv_analyst
import app.llm_client as llm_client
import app.ui_analyst as ui_analyst
from app.cv_analyst import _sections_from_markdown, add_skill_evidence, analyse_cv
from app.cv_generator import generate_cv_markdown
from app.schemas import StudentProfile
from app.ui_analyst import (
    ANALYSE_BUTTON_KEY,
    NOT_ANALYSED_CAPTION,
    STALE_CAPTION,
    display_groups,
    source_caption,
)
from app.vector_store import JobRecord

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_JOB = "graduate_ai_engineer"


def _rows(name: str) -> list[dict]:
    lines = (DATA_DIR / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


PROFILE_ROWS = {row["document_id"]: row for row in _rows("student_profiles.jsonl")}
JOBS = {row["document_id"]: row for row in _rows("jobs.jsonl")}


@pytest.fixture(autouse=True)
def _no_live_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """The analyst calls Gemini through app.llm_client, so stub it at the source."""

    def no_call(*_args, **_kwargs):
        raise AssertionError("live Gemini call attempted in test_ui_analyst")

    monkeypatch.setattr(llm_client, "get_gemini_client", lambda: None)
    monkeypatch.setattr(llm_client, "call_gemini_json", no_call)


@pytest.fixture
def analyse_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Count the panel's analyse_cv calls, still running the real analysis."""
    calls: list[tuple] = []
    real = ui_analyst.analyse_cv

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(ui_analyst, "analyse_cv", counting)
    return calls


class _GeminiStub:
    """Stands in for app.llm_client: echoes each issue with plain text."""

    def __init__(self, monkeypatch, error=None) -> None:
        self.calls: list[dict] = []
        self.error = error
        monkeypatch.setattr(llm_client, "get_gemini_client", lambda: "stub-client")
        monkeypatch.setattr(llm_client, "call_gemini_json", self)

    def __call__(self, client, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        payload = json.loads(kwargs["user_content"])
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


def _maya() -> StudentProfile:
    profile = StudentProfile.model_validate(PROFILE_ROWS["student_profile_01"])
    profile.is_confirmed = True
    return profile


def _sections(profile: StudentProfile, job_id: str = DEFAULT_JOB) -> dict[str, str]:
    job = JOBS[job_id]
    return _sections_from_markdown(generate_cv_markdown(profile, job, job["description"]))


def _selected_job(job_id: str = DEFAULT_JOB, with_description: bool = True) -> dict:
    """The job dict the app stores: a job match, optionally with its description."""
    job = JOBS[job_id]
    selected = {"document_id": job["document_id"], "title": job["title"]}
    if with_description:
        selected["description"] = job["description"]
    return selected


def _panel_script() -> None:
    from app.ui_analyst import render_analysis_panel

    render_analysis_panel()


def _panel(profile, sections, job) -> AppTest:
    at = AppTest.from_function(_panel_script, default_timeout=30)
    at.session_state["profile"] = profile
    at.session_state["cv_sections"] = sections
    at.session_state["selected_job"] = job
    return at.run()


def _press_analyse(at: AppTest) -> AppTest:
    return at.button(key=ANALYSE_BUTTON_KEY).click().run()


def _captions(at: AppTest) -> list[str]:
    return [caption.value for caption in at.caption]


def _shown_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + _captions(at)
    parts += [expander.label for expander in at.expander]
    return "\n".join(parts)


def test_nothing_is_analysed_before_the_button(analyse_calls: list[tuple]) -> None:
    maya = _maya()
    at = _panel(maya, _sections(maya), _selected_job())
    at.run()
    assert analyse_calls == []
    assert at.session_state["analysis_issues"] == []
    assert at.session_state["analysis_snapshot"] == {}
    assert at.session_state["analysis_diagnostics"] == {}
    assert NOT_ANALYSED_CAPTION in _captions(at)
    assert len(at.expander) == 0


def test_analyse_shows_maya_issues(analyse_calls: list[tuple]) -> None:
    maya = _maya()
    sections = _sections(maya)
    at = _press_analyse(_panel(maya, sections, _selected_job()))

    expected = analyse_cv(_maya(), sections, _selected_job())
    assert at.session_state["analysis_issues"] == [i.model_dump() for i in expected]
    assert len(analyse_calls) == 1
    markdown = [m.value for m in at.markdown]
    assert "**11 issues:** 0 high, 6 medium, 5 low" in markdown
    headings = [value for value in markdown if value.startswith("#### ")]
    assert headings == ["#### Experience", "#### Job ad", "#### Projects"]
    shown = _shown_text(at)
    for issue in expected:
        assert issue.why in shown
    no_key = "Explanations: deterministic templates (no Gemini key configured)."
    assert no_key in _captions(at)
    assert STALE_CAPTION not in _captions(at)


def test_exactly_one_analysis_per_press(analyse_calls: list[tuple]) -> None:
    maya = _maya()
    at = _press_analyse(_panel(maya, _sections(maya), _selected_job()))
    assert len(analyse_calls) == 1
    at.run()
    at.run()
    assert len(analyse_calls) == 1
    _press_analyse(at)
    assert len(analyse_calls) == 2


def test_editing_a_section_marks_the_analysis_out_of_date(
    analyse_calls: list[tuple],
) -> None:
    maya = _maya()
    sections = _sections(maya)
    at = _press_analyse(_panel(maya, sections, _selected_job()))
    shown_before = len(at.expander)

    at.session_state["cv_sections"] = {
        **sections,
        "Experience": sections["Experience"] + "\n- Trained new staff on the till.",
    }
    at.run()

    assert STALE_CAPTION in _captions(at)
    assert len(at.expander) == shown_before  # still shown, not hidden
    assert len(analyse_calls) == 1


def test_adding_evidence_marks_the_analysis_out_of_date(
    analyse_calls: list[tuple],
) -> None:
    maya = _maya()
    at = _press_analyse(_panel(maya, _sections(maya), _selected_job()))
    profile = at.session_state["profile"]
    add_skill_evidence(profile, "Flask", "flask_screenshot.png")
    at.session_state["profile"] = profile
    at.run()
    assert STALE_CAPTION in _captions(at)

    _press_analyse(at)
    assert STALE_CAPTION not in _captions(at)
    quotes = [issue["quote"] for issue in at.session_state["analysis_issues"]]
    assert "Flask" not in quotes and "SQLite" in quotes


def test_changing_the_job_marks_the_analysis_out_of_date() -> None:
    maya = _maya()
    at = _press_analyse(_panel(maya, _sections(maya), _selected_job()))
    at.session_state["selected_job"] = _selected_job("rag_agent_engineer")
    at.run()
    assert STALE_CAPTION in _captions(at)


def test_issues_are_grouped_by_section_and_entry() -> None:
    maya = _maya()
    sections = _sections(maya)
    issues = [i.model_dump() for i in analyse_cv(maya, sections, _selected_job())]
    groups = display_groups(issues, sections)

    assert [section for section, _ in groups] == ["Experience", "Job ad", "Projects"]
    experience, job_ad, projects = (items for _, items in groups)
    assert [label for label, _ in experience] == [
        "Medium · Vague · Handled customer questions, worked with a small team...",
        "Low · Job match · Handled customer questions, worked with a small team...",
    ]
    assert [label for label, _ in job_ad] == [
        "Medium · Job match · machine-learning fundamentals",
        "Medium · Job match · statistics",
        "Medium · Job match · data analysis",
        "Medium · Job match · LLMs",
        "Medium · Job match · RAG",
    ]
    assert [(label, [i["quote"] for i in members]) for label, members in projects] == [
        ("Low · Not shown · Team Course Planner: Flask, SQLite", ["Flask", "SQLite"]),
        (
            "Low · Not shown · AI Coursework Prototype: prompt design, JSON",
            ["prompt design", "JSON"],
        ),
    ]


def test_panel_shows_the_grouped_labels() -> None:
    maya = _maya()
    at = _press_analyse(_panel(maya, _sections(maya), _selected_job()))
    labels = [expander.label for expander in at.expander]
    assert "Low · Not shown · Team Course Planner: Flask, SQLite" in labels
    assert "Low · Not shown · AI Coursework Prototype: prompt design, JSON" in labels
    assert len(labels) == 9  # 11 issues, four of them in two shared expanders


def test_description_is_looked_up_when_the_job_has_none(
    monkeypatch: pytest.MonkeyPatch, analyse_calls: list[tuple]
) -> None:
    job = JOBS[DEFAULT_JOB]
    record = JobRecord(
        document_id=job["document_id"], title=job["title"], description=job["description"]
    )
    monkeypatch.setattr(ui_analyst, "load_jobs", lambda: [record])
    maya = _maya()
    at = _press_analyse(_panel(maya, _sections(maya), _selected_job(with_description=False)))

    [(_, _, passed_job)] = analyse_calls
    assert passed_job["description"] == job["description"]
    sections = {issue["section"] for issue in at.session_state["analysis_issues"]}
    assert "Job ad" in sections


def test_gemini_text_is_shown_with_its_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _GeminiStub(monkeypatch)
    maya = _maya()
    at = _press_analyse(_panel(maya, _sections(maya), _selected_job()))
    assert len(stub.calls) == 1
    assert "llm-written: a reader cannot see what this part shows." in _shown_text(at)
    assert (
        "Explanations: written by Gemini; every one passed the grounding and wording checks."
        in _captions(at)
    )
    at.run()
    assert len(stub.calls) == 1


def test_raw_error_details_are_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SdkError(Exception):
        code = 503

    error = llm_client.GeminiAPIError("Gemini generate_content failed: 503 UNAVAILABLE.")
    error.__cause__ = _SdkError()
    stub = _GeminiStub(monkeypatch, error=error)
    monkeypatch.setattr(cv_analyst.time, "sleep", lambda _seconds: None)
    maya = _maya()
    at = _press_analyse(_panel(maya, _sections(maya), _selected_job()))

    assert len(stub.calls) == 2  # the analyst's one retry on 503
    assert "Explanations: deterministic templates (Gemini call failed)." in _captions(at)
    shown = _shown_text(at)
    assert "generate_content" not in shown
    assert "503" not in shown


def test_panel_waits_for_a_cv(analyse_calls: list[tuple]) -> None:
    at = _panel(_maya(), {}, None)
    assert "Generate a CV for a job to analyse it." in _captions(at)
    assert len(at.button) == 0
    assert analyse_calls == []


def test_source_caption_names_short_reasons_only() -> None:
    diagnostics = {
        "detail": "Gemini generate_content failed: 400 something raw",
        "issues": {
            "a": {"source": "gemini", "reason": "", "detail": ""},
            "b": {"source": "template", "reason": "ungrounded", "detail": "why: Kubernetes"},
            "c": {"source": "template", "reason": "banned", "detail": "why: hire"},
        },
    }
    assert source_caption(diagnostics) == (
        "Explanations: written by Gemini for 1 of 3 issues; templates for the rest "
        "(Gemini text named something not in your profile (1); "
        "Gemini text broke a wording rule (1))."
    )
    assert source_caption({}) == ""
