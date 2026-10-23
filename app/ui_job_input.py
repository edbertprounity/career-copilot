"""Streamlit UI for parsing a custom job posting.

Area 3 contract
---------------
``render_job_input_panel()`` takes no arguments. On explicit approval it
appends a JobRecord-like dict to ``st.session_state["custom_jobs"]``:
``document_id`` starts with ``custom-``, plus ``title``, ``location``,
``job_type``, ``description``, and ``requirements``. Nothing is written to
``jobs.jsonl`` or ranking embeddings.

Pending review lives at ``parsed_job``. Parse errors at ``job_input_error``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import app.job_input as job_input
import app.llm_client as llm_client

CUSTOM_JOB_MATCH_KEY = "custom_job_match"
CUSTOM_JOBS_KEY = "custom_jobs"
PARSED_JOB_KEY = "parsed_job"
JOB_INPUT_ERROR_KEY = "job_input_error"

_WIDGET_TEXT = "job_input_text"
_WIDGET_FILES = "job_input_files"

_REVIEW_STRINGS = (
    ("Title", "job_title"),
    ("Company", "company"),
    ("Location", "location"),
    ("Employment type", "employment_type"),
    ("Summary", "raw_summary"),
)
_REVIEW_LISTS = (
    ("Key responsibilities", "key_responsibilities"),
    ("Qualifications and education", "qualifications_and_education"),
    ("Required skills", "required_skills"),
    ("Preferred skills", "preferred_skills"),
    ("Domain keywords", "domain_keywords"),
)


def _state(state: Any | None = None) -> Any:
    return st.session_state if state is None else state


def init_job_input_state(state: Any | None = None) -> None:
    """Ensure job-input keys exist. Does not overwrite an approved match."""
    store = _state(state)
    store.setdefault(PARSED_JOB_KEY, None)
    store.setdefault(CUSTOM_JOB_MATCH_KEY, None)
    store.setdefault(JOB_INPUT_ERROR_KEY, None)
    store.setdefault(CUSTOM_JOBS_KEY, [])


def clear_custom_job(state: Any | None = None) -> None:
    """Clear pending parse, errors, and approved custom jobs."""
    store = _state(state)
    store[PARSED_JOB_KEY] = None
    store[CUSTOM_JOB_MATCH_KEY] = None
    store[JOB_INPUT_ERROR_KEY] = None
    store[CUSTOM_JOBS_KEY] = []


def get_custom_job_match(state: Any | None = None) -> dict | None:
    store = _state(state)
    match = store.get(CUSTOM_JOB_MATCH_KEY)
    return match if isinstance(match, dict) else None


def _normalize_files(uploads: Any) -> list[tuple[bytes, str, str]]:
    files: list[tuple[bytes, str, str]] = []
    for item in uploads or []:
        if isinstance(item, tuple) and len(item) == 3:
            data, name, mime = item
            files.append((data, str(name or "upload"), str(mime or "")))
            continue
        data = item.getvalue() if hasattr(item, "getvalue") else bytes(item)
        name = getattr(item, "name", None) or "upload"
        mime = getattr(item, "type", None) or ""
        files.append((data, str(name), str(mime)))
    return files


def _student_error(exc: BaseException) -> str:
    text = str(exc).strip() or "the posting could not be parsed"
    if "fetching URLs" in text:
        return (
            "Paste the job-ad text from the page. Fetching URLs is not supported."
        )
    if text.startswith("Couldn't parse"):
        return text
    return f"Couldn't parse that job posting: {text}"


def submit_job_input(
    text: str | None,
    uploads: Any = None,
    *,
    client: Any = None,
    state: Any | None = None,
) -> None:
    """Parse pasted text or uploads into a pending review result.

    Does not approve the job. Does not call Gemini when input is empty.
    """
    store = _state(state)
    init_job_input_state(store)
    store[JOB_INPUT_ERROR_KEY] = None
    store[PARSED_JOB_KEY] = None

    files = _normalize_files(uploads)
    stripped = (text or "").strip()
    if not stripped and not files:
        store[JOB_INPUT_ERROR_KEY] = (
            "Paste a job advertisement or upload a screenshot or PDF."
        )
        return

    if client is None:
        client = llm_client.get_gemini_client()
    if client is None:
        store[JOB_INPUT_ERROR_KEY] = (
            "Job parsing needs a Gemini API key. Add GEMINI_API_KEY to your .env file."
        )
        return

    try:
        if files:
            parsed = job_input.parse_job_from_files(client, files)
        else:
            parsed = job_input.parse_job_description(client, stripped)
    except job_input.JobInputError as exc:
        store[JOB_INPUT_ERROR_KEY] = _student_error(exc)
        return

    store[PARSED_JOB_KEY] = parsed


def _area3_custom_job(parsed: dict, match: dict) -> dict:
    """JobRecord-like dict for ``st.session_state.custom_jobs``."""
    requirements = [
        str(item).strip()
        for item in (parsed.get("required_skills") or [])
        if str(item).strip()
    ]
    return {
        "document_id": match["document_id"],
        "title": match.get("title"),
        "location": match.get("location"),
        "job_type": match.get("job_type"),
        "description": match.get("description") or "",
        "requirements": requirements,
    }


def approve_parsed_job(state: Any | None = None) -> dict | None:
    """Convert the pending parse into an approved custom job (Area 3 list)."""
    store = _state(state)
    init_job_input_state(store)
    parsed = store.get(PARSED_JOB_KEY)
    if not isinstance(parsed, dict):
        store[JOB_INPUT_ERROR_KEY] = "Parse a job posting before using it."
        return None
    try:
        match = job_input.parsed_job_to_match(parsed)
    except job_input.JobInputError as exc:
        store[JOB_INPUT_ERROR_KEY] = _student_error(exc)
        return None
    custom = _area3_custom_job(parsed, match)
    jobs = [
        item
        for item in (store.get(CUSTOM_JOBS_KEY) or [])
        if isinstance(item, dict)
        and item.get("document_id") != custom["document_id"]
    ]
    jobs.append(custom)
    store[CUSTOM_JOBS_KEY] = jobs
    store[CUSTOM_JOB_MATCH_KEY] = match
    store[JOB_INPUT_ERROR_KEY] = None
    return match


def _render_review(parsed: dict) -> None:
    st.markdown("**Review parsed job**")
    st.caption("Check these fields. Nothing is used for your CV until you click Use this job.")
    for error in parsed.get("conversion_errors") or []:
        st.warning(str(error))
    for label, key in _REVIEW_STRINGS:
        value = parsed.get(key)
        display = str(value).strip() if value else "none stated"
        st.write(f"**{label}:** {display}")
    for label, key in _REVIEW_LISTS:
        values = [str(item).strip() for item in (parsed.get(key) or []) if str(item).strip()]
        if values:
            st.markdown(f"**{label}:**")
            for item in values:
                st.write(f"- {item}")
        else:
            st.write(f"**{label}:** none stated")


def render_job_input() -> None:
    """Render the custom job-input panel. Does not parse until the user submits."""
    init_job_input_state()
    st.subheader("Custom job posting")
    st.caption(
        "Paste the advertisement text, or upload a PNG/JPEG screenshot or a "
        "text-based PDF. A URL by itself is not enough — paste the text from "
        "the page. Uploads stay in memory and are not saved to disk."
    )
    st.text_area(
        "Job advertisement",
        key=_WIDGET_TEXT,
        height=180,
        placeholder="Paste the job description text (not just a URL).",
    )
    st.file_uploader(
        "Job ad screenshot or PDF",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        key=_WIDGET_FILES,
    )

    actions = st.columns(2)
    with actions[0]:
        parse_clicked = st.button("Parse job posting", key="job_input_submit")
    with actions[1]:
        clear_clicked = st.button("Clear custom job", key="job_input_clear")

    if clear_clicked:
        clear_custom_job()
    elif parse_clicked:
        submit_job_input(
            st.session_state.get(_WIDGET_TEXT),
            st.session_state.get(_WIDGET_FILES),
        )

    error = st.session_state.get(JOB_INPUT_ERROR_KEY)
    if error:
        st.error(error)

    parsed = st.session_state.get(PARSED_JOB_KEY)
    if isinstance(parsed, dict):
        _render_review(parsed)
        if st.button("Use this job", key="job_input_approve"):
            approve_parsed_job()

    jobs = st.session_state.get(CUSTOM_JOBS_KEY) or []
    if jobs:
        latest = jobs[-1] if isinstance(jobs[-1], dict) else {}
        title = latest.get("title") or "Custom job"
        document_id = latest.get("document_id") or ""
        extra = f" ({len(jobs)} saved)" if len(jobs) > 1 else ""
        st.success(f"Approved custom job: {title} (`{document_id}`){extra}.")


def render_job_input_panel() -> None:
    """Area 3 entry point: no arguments; reads and writes session state."""
    render_job_input()
