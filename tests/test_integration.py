"""Area 3 integration: session keys, reset, custom jobs, kill switch. No live Gemini."""

from __future__ import annotations

import pytest
import streamlit as st

import app.llm_client as llm_client
import streamlit_app as ui
from app.schemas import StudentProfile

CUSTOM = {
    "document_id": "custom-abc123",
    "title": "Data Analyst Intern",
    "location": "Sydney",
    "job_type": "Internship",
    "description": "Help the analytics team build dashboards.",
    "requirements": ["SQL", "Python", "Tableau"],
}


@pytest.fixture(autouse=True)
def _clean_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_init_state_adds_new_keys_as_empty_lists() -> None:
    ui._init_state()
    for key in ("analysis_issues", "evidence_records", "custom_jobs"):
        assert st.session_state[key] == []


def test_init_state_keeps_existing_values() -> None:
    st.session_state.custom_jobs = [CUSTOM]
    ui._init_state()
    assert st.session_state.custom_jobs == [CUSTOM]


def test_new_keys_do_not_share_one_list() -> None:
    ui._init_state()
    st.session_state.analysis_issues.append({"issue_id": "x"})
    assert st.session_state.evidence_records == []
    assert st.session_state.custom_jobs == []


def test_reset_session_clears_new_keys() -> None:
    ui._init_state()
    st.session_state.analysis_issues = [{"issue_id": "1"}]
    st.session_state.evidence_records = [{"claim": "c"}]
    st.session_state.custom_jobs = [CUSTOM]
    ui._reset_session()
    assert st.session_state.analysis_issues == []
    assert st.session_state.evidence_records == []
    assert st.session_state.custom_jobs == []


def test_job_description_reads_custom_job_before_jobs_file(monkeypatch) -> None:
    ui._init_state()
    st.session_state.custom_jobs = [CUSTOM]
    monkeypatch.setattr(ui, "load_jobs", lambda: pytest.fail("jobs.jsonl was read"))
    text = ui._job_description("custom-abc123")
    assert "analytics team" in text
    assert "SQL; Python; Tableau" in text


def test_job_description_falls_back_to_jobs_file(monkeypatch) -> None:
    class Job:
        document_id = "graduate_ai_engineer"
        description = "Build ML systems."

    ui._init_state()
    monkeypatch.setattr(ui, "load_jobs", lambda: [Job()])
    assert ui._job_description("graduate_ai_engineer") == "Build ML systems."
    assert ui._job_description("missing") == ""


def test_custom_job_card_uses_fallback_without_client_and_caches() -> None:
    ui._init_state()
    st.session_state.profile = StudentProfile(is_confirmed=True)
    custom = dict(CUSTOM)
    card = ui._custom_job_card(custom)
    assert card["gap_analysis"] == ui.GAP_ANALYSIS_FALLBACK
    assert card["document_id"] == "custom-abc123"
    assert card["supporting_excerpt"] == "SQL; Python; Tableau"
    assert custom["gap_analysis"] == ui.GAP_ANALYSIS_FALLBACK  # cached on the job


def test_custom_job_card_calls_gap_analysis_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(ui.llm_client, "get_gemini_client", lambda: object())
    monkeypatch.setattr(
        ui, "generate_gap_analysis", lambda *a, **k: calls.append(a) or "Gap text."
    )
    ui._init_state()
    custom = dict(CUSTOM)
    ui._custom_job_card(custom)
    ui._custom_job_card(custom)
    assert len(calls) == 1
    assert custom["gap_analysis"] == "Gap text."


def test_select_custom_job_builds_cv_and_clears_old_issues(confirmed_profile, monkeypatch) -> None:
    ui._init_state()
    st.session_state.profile = confirmed_profile
    st.session_state.custom_jobs = [CUSTOM]
    st.session_state.analysis_issues = [{"issue_id": "old"}]
    monkeypatch.setattr(ui.llm_client, "get_gemini_client", lambda: None)
    card = ui._custom_job_card(st.session_state.custom_jobs[0])
    ui._select_job(card)
    assert st.session_state.selected_job["document_id"] == "custom-abc123"
    assert "Education" in st.session_state.cv_sections
    assert st.session_state.analysis_issues == []
    assert st.session_state.pdf_bytes


def test_kill_switch_blocks_parts_call_when_present() -> None:
    if not hasattr(llm_client, "call_gemini_parts_json"):
        pytest.skip("call_gemini_parts_json not merged yet (Area 2)")
    with pytest.raises(AssertionError, match="live Gemini call"):
        llm_client.call_gemini_parts_json(None, "sys", [], {})


def test_kill_switch_blocks_json_call() -> None:
    with pytest.raises(AssertionError, match="live Gemini call"):
        llm_client.call_gemini_json(None, "sys", "hi")
