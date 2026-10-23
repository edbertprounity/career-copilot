"""Streamlit panel for reviewing uploaded evidence before it touches the profile.

``render_evidence_panel()`` takes no arguments. Extraction writes a pending
review payload; nothing is added to the profile until the student applies a
finding. Applied skills go through ``add_skill_evidence`` and are also
appended to ``st.session_state["evidence_records"]``. Uploads stay in memory.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import app.llm_client as llm_client
from app.cv_analyst import add_skill_evidence
from app.evidence import extract_evidence_from_files

EVIDENCE_RECORDS_KEY = "evidence_records"
PENDING_KEY = "pending_evidence"
ERROR_KEY = "evidence_error"
_WIDGET_FILES = "evidence_files"
_CLAIM_TYPES = (
    ("skills", "skill", "Skill"),
    ("work_experience", "role", "Experience"),
    ("education", "degree", "Education"),
    ("projects", "title", "Project"),
    ("certifications", "name", "Certification"),
)


def _init_evidence_state() -> None:
    st.session_state.setdefault(EVIDENCE_RECORDS_KEY, [])
    st.session_state.setdefault(PENDING_KEY, None)
    st.session_state.setdefault(ERROR_KEY, None)


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


def _source_for(item: dict) -> str:
    filename = str(item.get("filename") or "upload").strip() or "upload"
    excerpt = str(item.get("excerpt") or "").strip()
    if excerpt:
        return f"{filename}: {excerpt}"
    return filename


def _claim_rows(extracted: dict) -> list[dict]:
    rows: list[dict] = []
    for group, field, label in _CLAIM_TYPES:
        for item in extracted.get(group) or []:
            if not isinstance(item, dict):
                continue
            claim = str(item.get(field) or "").strip()
            if not claim:
                continue
            rows.append(
                {
                    "kind": group,
                    "label": label,
                    "claim": claim,
                    "skill": claim if group == "skills" else "",
                    "source": _source_for(item),
                    "filename": str(item.get("filename") or ""),
                    "excerpt": str(item.get("excerpt") or ""),
                    "page": item.get("page"),
                }
            )
    return rows


def submit_evidence_files(uploads: Any = None, *, client: Any = None) -> None:
    """Extract a pending review payload. Does not write the profile."""
    _init_evidence_state()
    st.session_state[ERROR_KEY] = None
    st.session_state[PENDING_KEY] = None
    files = _normalize_files(uploads)
    if not files:
        st.session_state[ERROR_KEY] = "Upload a screenshot or PDF of your evidence first."
        return
    if client is None:
        client = llm_client.get_gemini_client()
    if client is None:
        st.session_state[ERROR_KEY] = (
            "Evidence extraction needs a Gemini API key. Add GEMINI_API_KEY to your .env file."
        )
        return
    extracted = extract_evidence_from_files(client, files)
    st.session_state[PENDING_KEY] = extracted
    errors = [str(item) for item in (extracted.get("errors") or []) if str(item).strip()]
    if errors and not _claim_rows(extracted):
        st.session_state[ERROR_KEY] = errors[0]


def apply_skill_finding(row: dict) -> dict:
    """Copy one skill finding onto the confirmed profile after student approval."""
    _init_evidence_state()
    profile = st.session_state.get("profile")
    if profile is None:
        return {"status": "error", "reason": "no profile"}
    skill = str(row.get("skill") or row.get("claim") or "").strip()
    source = str(row.get("source") or row.get("filename") or "").strip()
    result = add_skill_evidence(profile, skill, source)
    if result.get("status") == "error":
        return result
    record = {
        "claim": skill,
        "skill": skill,
        "source": source,
        "filename": str(row.get("filename") or ""),
        "excerpt": str(row.get("excerpt") or ""),
    }
    records = list(st.session_state.get(EVIDENCE_RECORDS_KEY) or [])
    if record not in records:
        records.append(record)
        st.session_state[EVIDENCE_RECORDS_KEY] = records
    return result


def render_evidence_panel() -> None:
    """Area 3 entry point: review uploads, then apply skills on request."""
    _init_evidence_state()
    st.subheader("Evidence")
    st.caption(
        "Upload a certificate, transcript, or screenshot. Findings stay in this "
        "review until you apply a skill. Files are not saved to disk."
    )
    st.file_uploader(
        "Evidence screenshot or PDF",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        key=_WIDGET_FILES,
    )
    if st.button("Extract evidence", key="evidence_extract"):
        submit_evidence_files(st.session_state.get(_WIDGET_FILES))

    error = st.session_state.get(ERROR_KEY)
    if error:
        st.error(error)

    pending = st.session_state.get(PENDING_KEY)
    if not isinstance(pending, dict):
        records = st.session_state.get(EVIDENCE_RECORDS_KEY) or []
        if records:
            st.caption(f"{len(records)} evidence record(s) applied this session.")
        return

    for warning in pending.get("uncertain_findings") or []:
        text = str(warning).strip()
        if text:
            st.warning(text)
    for item in pending.get("errors") or []:
        text = str(item).strip()
        if text:
            st.caption(text)

    rows = _claim_rows(pending)
    if not rows:
        st.caption("No explicit findings to apply.")
        return

    st.markdown("**Review findings**")
    for index, row in enumerate(rows):
        with st.container(border=True):
            st.write(f"**{row['label']}:** {row['claim']}")
            if row.get("excerpt"):
                st.caption(row["excerpt"])
            if row["kind"] == "skills":
                if st.button("Add this skill to my profile", key=f"evidence_apply_{index}"):
                    result = apply_skill_finding(row)
                    if result.get("status") == "error":
                        st.session_state[ERROR_KEY] = str(
                            result.get("reason") or "could not add that skill"
                        )
                    st.rerun()
            else:
                st.caption("Shown for review only. Skills can be added to the profile.")
