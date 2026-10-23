"""Streamlit custom-job wiring. No live Gemini; no corpus writes."""

from __future__ import annotations

import streamlit_app as ui
from app.ui_job_input import CUSTOM_JOB_MATCH_KEY, CUSTOM_JOBS_KEY
from streamlit.testing.v1 import AppTest

CUSTOM = {
    "document_id": "custom-a79ebb7d5e7aa5cd",
    "title": "Graduate AI Engineer",
    "location": "Sydney NSW",
    "job_type": "Full time",
    "description": "Required skills: Python, PyTorch\n\nKey responsibilities:\n- Build models",
    "requirements": ["Python", "PyTorch"],
}

CUSTOM_DESCRIPTION = (
    "Required skills: Python, PyTorch\n\n"
    "Key responsibilities:\n- Build models"
)

RANKED = {
    "document_id": "graduate_ai_engineer",
    "title": "Graduate AI Engineer",
    "similarity_score": 0.81,
    "supporting_excerpt": "Develop and evaluate machine-learning models",
    "gap_analysis": "Gap analysis unavailable for this match.",
}


def test_render_jobs_keeps_ranked_matches_separate() -> None:
    def _app() -> None:
        from streamlit_app import _init_state, _render_jobs

        _init_state()
        _render_jobs()

    at = AppTest.from_function(_app, default_timeout=30)
    at.session_state["job_matches"] = [dict(RANKED)]
    at.session_state[CUSTOM_JOBS_KEY] = [dict(CUSTOM)]
    at.session_state["profile"] = __import__(
        "app.schemas", fromlist=["StudentProfile"]
    ).StudentProfile(is_confirmed=True)
    at.run()
    assert not at.exception
    assert at.session_state["job_matches"] == [RANKED]
    subheads = [item.value for item in at.subheader]
    assert "Matching jobs" in subheads
    assert "Your job ads" in subheads
    assert at.button(key=f"select_{CUSTOM['document_id']}") is not None
    assert at.button(key=f"select_{RANKED['document_id']}") is not None


def test_select_job_prefers_match_description(
    monkeypatch, confirmed_profile
) -> None:
    captured: dict = {}

    def fake_markdown(profile, job, description, professional_summary=None):
        captured["description"] = description
        captured["title"] = job.get("title")
        return "# Test\n\n## Education\nUTS\n"

    monkeypatch.setattr(ui, "generate_cv_markdown", fake_markdown)
    monkeypatch.setattr(ui, "_store_cv_pdf", lambda _markdown: None)
    monkeypatch.setattr(ui.llm_client, "get_gemini_client", lambda: None)

    def _app() -> None:
        import streamlit as st
        from streamlit_app import _init_state, _select_job

        _init_state()
        job = {
            "document_id": "custom-a79ebb7d5e7aa5cd",
            "title": "Graduate AI Engineer",
            "description": (
                "Required skills: Python, PyTorch\n\n"
                "Key responsibilities:\n- Build models"
            ),
        }
        if st.button("select custom", key="select_custom_test"):
            _select_job(job)

    at = AppTest.from_function(_app, default_timeout=30)
    at.session_state["profile"] = confirmed_profile
    at.run()
    at.button(key="select_custom_test").click().run()
    assert not at.exception
    assert captured["description"] == CUSTOM_DESCRIPTION
    assert captured["title"] == CUSTOM["title"]
    assert at.session_state["selected_job"]["document_id"] == CUSTOM["document_id"]
    assert at.session_state["analysis_issues"] == []


def test_select_job_falls_back_to_corpus_when_description_missing(
    monkeypatch, confirmed_profile
) -> None:
    captured: dict = {}

    def fake_markdown(profile, job, description, professional_summary=None):
        captured["description"] = description
        return "# Test\n\n## Education\nUTS\n"

    monkeypatch.setattr(ui, "generate_cv_markdown", fake_markdown)
    monkeypatch.setattr(ui, "_store_cv_pdf", lambda _markdown: None)
    monkeypatch.setattr(ui.llm_client, "get_gemini_client", lambda: None)
    monkeypatch.setattr(
        ui,
        "_job_description",
        lambda document_id: f"corpus text for {document_id}",
    )

    def _app() -> None:
        import streamlit as st
        from streamlit_app import _init_state, _select_job

        _init_state()
        job = {
            "document_id": "graduate_ai_engineer",
            "title": "Graduate AI Engineer",
            "similarity_score": 0.81,
        }
        if st.button("select ranked", key="select_ranked_test"):
            _select_job(job)

    at = AppTest.from_function(_app, default_timeout=30)
    at.session_state["profile"] = confirmed_profile
    at.run()
    at.button(key="select_ranked_test").click().run()
    assert not at.exception
    assert captured["description"] == "corpus text for graduate_ai_engineer"


def test_reset_session_clears_custom_jobs() -> None:
    def _app() -> None:
        import streamlit as st
        from streamlit_app import _init_state, _reset_session

        _init_state()
        if st.button("Start New Profile", key="reset_custom_test"):
            _reset_session()

    at = AppTest.from_function(_app, default_timeout=30)
    at.session_state[CUSTOM_JOB_MATCH_KEY] = {"document_id": CUSTOM["document_id"]}
    at.session_state[CUSTOM_JOBS_KEY] = [dict(CUSTOM)]
    at.session_state["parsed_job"] = {"job_title": "Graduate AI Engineer"}
    at.run()
    assert at.session_state[CUSTOM_JOBS_KEY][0]["document_id"] == CUSTOM["document_id"]
    at.button(key="reset_custom_test").click().run()
    assert at.session_state[CUSTOM_JOB_MATCH_KEY] is None
    assert at.session_state[CUSTOM_JOBS_KEY] == []
    assert at.session_state["parsed_job"] is None
    assert at.session_state["job_matches"] == []
    assert at.session_state["analysis_issues"] == []


def test_main_renders_job_input_and_custom_card() -> None:
    def _app() -> None:
        from streamlit_app import _init_state, _render_jobs
        from app.ui_job_input import render_job_input_panel

        _init_state()
        render_job_input_panel()
        _render_jobs()

    at = AppTest.from_function(_app, default_timeout=30)
    at.session_state["job_matches"] = [dict(RANKED)]
    at.session_state[CUSTOM_JOBS_KEY] = [dict(CUSTOM)]
    at.session_state["profile"] = __import__(
        "app.schemas", fromlist=["StudentProfile"]
    ).StudentProfile(is_confirmed=True)
    at.run()
    assert not at.exception
    assert any("Custom job posting" in item.value for item in at.subheader)
    assert at.session_state["job_matches"] == [RANKED]
    assert at.button(key=f"select_{CUSTOM['document_id']}") is not None
    assert at.button(key=f"select_{RANKED['document_id']}") is not None
