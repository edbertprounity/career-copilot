"""Chat helpers for attachments and rich assistant bubbles.

Uploads stay in memory. Facts extracted from files are shown for review and
passed to the profile agent as text; they are not written until a tool or
the student applies them.
"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

IMAGE_MIME = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
CHAT_FILE_TYPES = ["png", "jpg", "jpeg", "webp", "pdf"]
JOB_HINT_RE = re.compile(
    r"\b(job ad|job posting|job description|vacancy|position description|"
    r"role description|this (?:job|role|posting)|key responsibilities|"
    r"required skills|we are (?:looking|hiring)|apply now)\b",
    re.I,
)
EVIDENCE_HINT_RE = re.compile(
    r"\b(transcript|certificate|certification|diploma|resume|cv|portfolio|"
    r"linkedin|evidence|my (?:degree|results|grades))\b",
    re.I,
)


def split_chat_prompt(prompt: Any) -> tuple[str, list]:
    """Return (text, uploaded files) from ``st.chat_input``."""
    if prompt is None:
        return "", []
    if isinstance(prompt, str):
        return prompt.strip(), []
    text = str(getattr(prompt, "text", "") or "").strip()
    files = list(getattr(prompt, "files", None) or [])
    return text, files


def normalize_uploads(uploads: Any) -> list[tuple[bytes, str, str]]:
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


def attachment_records(files: list[tuple[bytes, str, str]]) -> list[dict]:
    records = []
    for data, name, mime in files:
        kind = "image" if (mime or "").lower() in IMAGE_MIME else "file"
        record = {"name": name, "mime": mime, "kind": kind}
        if kind == "image":
            record["data"] = data
        records.append(record)
    return records


def classify_chat_intent(
    text: str,
    files: list[tuple[bytes, str, str]],
    *,
    profile_confirmed: bool,
    has_selected_job: bool,
    chat_locked: bool,
) -> str:
    """``agent``, ``job``, ``evidence``, or ``locked``."""
    stripped = (text or "").strip()
    jobish = bool(JOB_HINT_RE.search(stripped))
    evidenceish = bool(EVIDENCE_HINT_RE.search(stripped))
    if files:
        if evidenceish:
            return "evidence"
        if jobish:
            return "job"
        if has_selected_job:
            return "evidence"
        if profile_confirmed:
            return "job"
        return "evidence"
    if jobish or (profile_confirmed and _looks_like_job_text(stripped)):
        return "job"
    if chat_locked:
        return "locked"
    return "agent"


def _looks_like_job_text(text: str) -> bool:
    if len(text) < 280:
        return False
    markers = (
        "responsibilit",
        "requirement",
        "qualification",
        "about the role",
        "what you will",
        "you will",
    )
    lower = text.lower()
    return sum(1 for marker in markers if marker in lower) >= 2


def compose_attachment_note(
    text: str,
    files: list[tuple[bytes, str, str]],
    extracted: dict | None,
) -> str:
    """Text the profile agent sees when the student attached files."""
    names = ", ".join(name for _data, name, _mime in files if name) or "upload"
    lines = [
        f"The student attached file(s): {names}.",
        "Facts extracted from the attachments follow. Use only these facts; "
        "do not invent employers, skills, dates, or results.",
    ]
    facts = _extraction_lines(extracted or {})
    if facts:
        lines.extend(facts)
    else:
        lines.append("- no explicit facts were extracted")
    if text:
        lines.append(f"Student message: {text}")
    else:
        lines.append("Student message: please read my attached file.")
    return "\n".join(lines)


def _extraction_lines(extracted: dict) -> list[str]:
    lines: list[str] = []
    for item in extracted.get("skills") or []:
        skill = str((item or {}).get("skill") or "").strip()
        if skill:
            lines.append(f"- skill: {skill}")
    for item in extracted.get("education") or []:
        degree = str((item or {}).get("degree") or "").strip()
        institution = str((item or {}).get("institution") or "").strip()
        if degree or institution:
            lines.append(f"- education: {degree} {institution}".strip())
    for item in extracted.get("work_experience") or []:
        role = str((item or {}).get("role") or "").strip()
        organisation = str((item or {}).get("organisation") or "").strip()
        if role or organisation:
            lines.append(f"- experience: {role} at {organisation}".strip())
    for item in extracted.get("projects") or []:
        title = str((item or {}).get("title") or "").strip()
        if title:
            lines.append(f"- project: {title}")
    for item in extracted.get("certifications") or []:
        name = str((item or {}).get("name") or "").strip()
        if name:
            lines.append(f"- certification: {name}")
    for item in extracted.get("uncertain_findings") or []:
        note = str(item).strip()
        if note:
            lines.append(f"- uncertain: {note}")
    return lines


def render_chat_item(item: dict) -> None:
    role = item.get("role", "user")
    if role not in ("user", "assistant"):
        role = "user"
    with st.chat_message(role):
        content = str(item.get("content") or "")
        if content:
            st.write(content)
        for attachment in item.get("attachments") or []:
            _render_attachment(attachment)
        for block in item.get("blocks") or []:
            _render_block(block)


def _render_attachment(attachment: dict) -> None:
    name = str(attachment.get("name") or "upload")
    if attachment.get("kind") == "image" and attachment.get("data"):
        st.image(attachment["data"], caption=name)
        return
    st.caption(f"Attached: {name}")


def _render_block(block: dict) -> None:
    kind = str(block.get("type") or "")
    if kind == "evidence":
        _render_evidence_block(block.get("extracted") or {})
        return
    if kind == "parsed_job":
        parsed = block.get("parsed") or {}
        st.markdown("**Parsed job ad**")
        title = parsed.get("job_title") or parsed.get("title") or "Custom job"
        st.write(title)
        if parsed.get("company"):
            st.caption(str(parsed["company"]))
        skills = parsed.get("required_skills") or []
        if skills:
            st.write("Required: " + ", ".join(str(item) for item in skills[:8]))
        return
    if kind == "job_cards":
        st.markdown("**Matching jobs**")
        for job in block.get("jobs") or []:
            title = job.get("title") or job.get("document_id") or "Job"
            excerpt = job.get("supporting_excerpt") or ""
            st.markdown(f"**{title}**")
            if excerpt:
                st.write(excerpt)
        return
    if kind == "cv":
        markdown = str(block.get("markdown") or "").strip()
        if markdown:
            st.markdown(markdown)
        pdf = block.get("pdf_bytes")
        if pdf:
            st.download_button(
                "Download this CV PDF",
                data=pdf,
                file_name="career_copilot_cv.pdf",
                mime="application/pdf",
                key=str(block.get("download_key") or "chat_cv_pdf"),
            )
        return
    if kind == "image" and block.get("data"):
        st.image(block["data"], caption=str(block.get("caption") or ""))


def _render_evidence_block(extracted: dict) -> None:
    rows = []
    for group, field, label in (
        ("skills", "skill", "Skill"),
        ("education", "degree", "Education"),
        ("work_experience", "role", "Experience"),
        ("projects", "title", "Project"),
        ("certifications", "name", "Certification"),
    ):
        for item in extracted.get(group) or []:
            claim = str((item or {}).get(field) or "").strip()
            if claim:
                rows.append(f"- **{label}:** {claim}")
    if rows:
        st.markdown("**From your attachment**")
        st.markdown("\n".join(rows))
    for warning in extracted.get("uncertain_findings") or []:
        text = str(warning).strip()
        if text:
            st.warning(text)
    for error in extracted.get("errors") or []:
        text = str(error).strip()
        if text:
            st.caption(text)
