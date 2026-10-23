"""Custom job-input UI tests. No live Gemini calls."""

from __future__ import annotations

import struct
import zlib

import pytest
from streamlit.testing.v1 import AppTest

import app.job_input as job_input
import app.ui_job_input as ui
from app.job_input import JobInputError
from app.ui_job_input import (
    CUSTOM_JOB_MATCH_KEY,
    CUSTOM_JOBS_KEY,
    JOB_INPUT_ERROR_KEY,
    PARSED_JOB_KEY,
    approve_parsed_job,
    clear_custom_job,
    init_job_input_state,
    submit_job_input,
)


PARSED_PAYLOAD = {
    "job_title": "Graduate AI Engineer",
    "company": "Example Labs",
    "location": "Sydney NSW",
    "employment_type": "Full time",
    "required_skills": ["Python", "PyTorch"],
    "preferred_skills": ["Docker"],
    "key_responsibilities": ["Build and evaluate machine-learning models"],
    "qualifications_and_education": [],
    "domain_keywords": ["Python", "PyTorch", "machine-learning"],
    "raw_summary": "Graduate AI Engineer role in Sydney using Python and PyTorch.",
}


def _png_bytes() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"\x00" + b"\xff\x00\x00" * 4
    ihdr = struct.pack(">IIBBBBB", 4, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _app() -> None:
    from app.ui_job_input import render_job_input

    render_job_input()


def _ss(at: AppTest, key: str):
    try:
        return at.session_state[key]
    except KeyError:
        return None


def _stub_parser(monkeypatch: pytest.MonkeyPatch, payload: dict | None = None) -> None:
    result = dict(payload or PARSED_PAYLOAD)
    monkeypatch.setattr(ui.llm_client, "get_gemini_client", lambda: object())
    monkeypatch.setattr(
        job_input, "parse_job_description", lambda *_args, **_kwargs: dict(result)
    )
    monkeypatch.setattr(
        job_input, "parse_job_from_files", lambda *_args, **_kwargs: dict(result)
    )


def test_init_does_not_call_gemini_or_approve() -> None:
    state: dict = {}
    init_job_input_state(state)
    assert state[PARSED_JOB_KEY] is None
    assert state[CUSTOM_JOB_MATCH_KEY] is None
    assert state[JOB_INPUT_ERROR_KEY] is None


def test_empty_submit_does_not_call_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("parser called without input")

    monkeypatch.setattr(job_input, "parse_job_description", boom)
    monkeypatch.setattr(job_input, "parse_job_from_files", boom)
    state: dict = {}
    submit_job_input("  ", [], client=object(), state=state)
    assert "Paste a job advertisement" in state[JOB_INPUT_ERROR_KEY]
    assert state[PARSED_JOB_KEY] is None
    assert state[CUSTOM_JOB_MATCH_KEY] is None


def test_submit_pasted_text_requires_explicit_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        job_input, "parse_job_description", lambda *_args, **_kwargs: dict(PARSED_PAYLOAD)
    )
    state: dict = {}
    submit_job_input(
        "Graduate AI Engineer at Example Labs. Python.",
        [],
        client=object(),
        state=state,
    )
    assert state[PARSED_JOB_KEY]["job_title"] == "Graduate AI Engineer"
    assert state[CUSTOM_JOB_MATCH_KEY] is None

    match = approve_parsed_job(state)
    assert match is not None
    assert state[CUSTOM_JOB_MATCH_KEY]["title"] == "Graduate AI Engineer"
    assert state[CUSTOM_JOB_MATCH_KEY]["document_id"].startswith("custom-")
    assert state[CUSTOM_JOBS_KEY][0]["document_id"].startswith("custom-")
    assert state[CUSTOM_JOBS_KEY][0]["requirements"] == ["Python", "PyTorch"]
    assert "Python" in state[CUSTOM_JOB_MATCH_KEY]["description"]


def test_submit_supported_image_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def stub(_client, files):
        captured["files"] = files
        return dict(PARSED_PAYLOAD)

    monkeypatch.setattr(job_input, "parse_job_from_files", stub)
    state: dict = {}
    submit_job_input(
        "",
        [(_png_bytes(), "ad.png", "image/png")],
        client=object(),
        state=state,
    )
    assert captured["files"][0][1] == "ad.png"
    assert state[PARSED_JOB_KEY]["job_title"] == "Graduate AI Engineer"
    assert state[CUSTOM_JOB_MATCH_KEY] is None


def test_parser_error_is_stored_for_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(*_args, **_kwargs):
        raise JobInputError("job parse failed: upstream")

    monkeypatch.setattr(job_input, "parse_job_description", stub)
    state: dict = {}
    submit_job_input("A genuine job advertisement.", [], client=object(), state=state)
    assert "Couldn't parse that job posting" in state[JOB_INPUT_ERROR_KEY]
    assert state[PARSED_JOB_KEY] is None
    assert state[CUSTOM_JOB_MATCH_KEY] is None


def test_url_only_error_is_actionable() -> None:
    state: dict = {}
    submit_job_input(
        "https://example.com/jobs/ai-engineer",
        [],
        client=object(),
        state=state,
    )
    assert "Paste the job-ad text" in state[JOB_INPUT_ERROR_KEY]
    assert "Fetching URLs is not supported" in state[JOB_INPUT_ERROR_KEY]
    assert state[PARSED_JOB_KEY] is None


def test_mixed_uploads_keep_parse_and_conversion_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(_client, files):
        payload = dict(PARSED_PAYLOAD)
        payload["conversion_errors"] = ["bad.pdf: not a readable PDF"]
        return payload

    monkeypatch.setattr(job_input, "parse_job_from_files", stub)
    state: dict = {}
    submit_job_input(
        "",
        [
            (_png_bytes(), "ok.png", "image/png"),
            (b"nope", "bad.pdf", "application/pdf"),
        ],
        client=object(),
        state=state,
    )
    parsed = state[PARSED_JOB_KEY]
    assert parsed["job_title"] == "Graduate AI Engineer"
    assert any("not a readable PDF" in item for item in parsed["conversion_errors"])
    assert state[CUSTOM_JOB_MATCH_KEY] is None
    approve_parsed_job(state)
    assert CUSTOM_JOB_MATCH_KEY in state
    assert "conversion_errors" not in state[CUSTOM_JOB_MATCH_KEY]


def test_approved_match_survives_rerun_init() -> None:
    monkeypatch_state: dict = {
        PARSED_JOB_KEY: dict(PARSED_PAYLOAD),
        CUSTOM_JOB_MATCH_KEY: None,
        JOB_INPUT_ERROR_KEY: None,
    }
    match = approve_parsed_job(monkeypatch_state)
    init_job_input_state(monkeypatch_state)
    init_job_input_state(monkeypatch_state)
    assert monkeypatch_state[CUSTOM_JOB_MATCH_KEY] == match
    assert monkeypatch_state[CUSTOM_JOB_MATCH_KEY]["title"] == "Graduate AI Engineer"


def test_clear_custom_job_removes_approved_match() -> None:
    state: dict = {
        PARSED_JOB_KEY: dict(PARSED_PAYLOAD),
        CUSTOM_JOB_MATCH_KEY: {"title": "Graduate AI Engineer"},
        JOB_INPUT_ERROR_KEY: "stale",
    }
    clear_custom_job(state)
    assert state[PARSED_JOB_KEY] is None
    assert state[CUSTOM_JOB_MATCH_KEY] is None
    assert state[JOB_INPUT_ERROR_KEY] is None
    assert state[CUSTOM_JOBS_KEY] == []


def test_replacing_custom_job_requires_new_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [dict(PARSED_PAYLOAD)]

    def stub(*_args, **_kwargs):
        return dict(payloads[-1])

    monkeypatch.setattr(job_input, "parse_job_description", stub)
    state: dict = {}
    submit_job_input("first posting", [], client=object(), state=state)
    approve_parsed_job(state)
    first_id = state[CUSTOM_JOB_MATCH_KEY]["document_id"]

    replacement = dict(PARSED_PAYLOAD)
    replacement["job_title"] = "Data Analyst"
    payloads.append(replacement)
    submit_job_input("second posting", [], client=object(), state=state)
    assert state[PARSED_JOB_KEY]["job_title"] == "Data Analyst"
    assert state[CUSTOM_JOB_MATCH_KEY]["document_id"] == first_id

    approve_parsed_job(state)
    assert state[CUSTOM_JOB_MATCH_KEY]["title"] == "Data Analyst"
    assert state[CUSTOM_JOB_MATCH_KEY]["document_id"] != first_id


def test_missing_client_does_not_call_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("parser called without client")

    monkeypatch.setattr(job_input, "parse_job_description", boom)
    monkeypatch.setattr(ui.llm_client, "get_gemini_client", lambda: None)
    state: dict = {}
    submit_job_input("A genuine job advertisement.", [], state=state)
    assert "GEMINI_API_KEY" in state[JOB_INPUT_ERROR_KEY]
    assert state[PARSED_JOB_KEY] is None


def test_render_job_input_without_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("live Gemini call attempted")

    monkeypatch.setattr(job_input, "parse_job_description", boom)
    monkeypatch.setattr(job_input, "parse_job_from_files", boom)
    at = AppTest.from_function(_app, default_timeout=30)
    at.run()
    assert not at.exception
    assert _ss(at, CUSTOM_JOB_MATCH_KEY) is None
    assert any("Custom job posting" in item.value for item in at.subheader)
    labels = [item.label for item in at.button]
    assert "Parse job posting" in labels
    assert "Use this job" not in labels


def test_render_shows_parser_error_on_empty_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("parser called without input")

    monkeypatch.setattr(job_input, "parse_job_description", boom)
    at = AppTest.from_function(_app, default_timeout=30)
    at.run()
    at.button(key="job_input_submit").click().run()
    assert not at.exception
    assert at.error
    assert "Paste a job advertisement" in at.error[0].value
    assert _ss(at, CUSTOM_JOB_MATCH_KEY) is None


def test_render_approve_stores_custom_job_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_parser(monkeypatch)
    at = AppTest.from_function(_app, default_timeout=30)
    at.run()
    at.text_area(key="job_input_text").set_value(
        "Graduate AI Engineer at Example Labs. Python."
    ).run()
    at.button(key="job_input_submit").click().run()
    assert _ss(at, PARSED_JOB_KEY)["job_title"] == "Graduate AI Engineer"
    assert _ss(at, CUSTOM_JOB_MATCH_KEY) is None
    assert any("Review parsed job" in item.value for item in at.markdown)
    at.button(key="job_input_approve").click().run()
    match = _ss(at, CUSTOM_JOB_MATCH_KEY)
    assert match["title"] == "Graduate AI Engineer"
    at.run()
    assert _ss(at, CUSTOM_JOB_MATCH_KEY)["document_id"] == match["document_id"]


def test_render_clear_removes_approved_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_parser(monkeypatch)
    at = AppTest.from_function(_app, default_timeout=30)
    at.run()
    at.text_area(key="job_input_text").set_value("Graduate AI Engineer Python.").run()
    at.button(key="job_input_submit").click().run()
    at.button(key="job_input_approve").click().run()
    assert _ss(at, CUSTOM_JOB_MATCH_KEY)["title"] == "Graduate AI Engineer"
    at.button(key="job_input_clear").click().run()
    assert _ss(at, CUSTOM_JOB_MATCH_KEY) is None
    assert _ss(at, PARSED_JOB_KEY) is None


def test_render_shows_conversion_errors_with_parsed_fields() -> None:
    payload = dict(PARSED_PAYLOAD)
    payload["conversion_errors"] = ["bad.pdf: not a readable PDF"]
    at = AppTest.from_function(_app, default_timeout=30)
    at.session_state[PARSED_JOB_KEY] = payload
    at.session_state[CUSTOM_JOB_MATCH_KEY] = None
    at.session_state[JOB_INPUT_ERROR_KEY] = None
    at.run()
    assert not at.exception
    assert at.warning
    assert "not a readable PDF" in at.warning[0].value
    markdown = " ".join(item.value or "" for item in at.markdown)
    assert "Review parsed job" in markdown
    assert "Graduate AI Engineer" in markdown or any(
        "Graduate AI Engineer" in (item.value or "") for item in at.markdown
    )
    assert _ss(at, CUSTOM_JOB_MATCH_KEY) is None
    assert "Use this job" in [item.label for item in at.button]
