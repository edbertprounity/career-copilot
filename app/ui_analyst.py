"""Streamlit panel for the CV analyst.

The panel owns its session state. analyse_cv runs only when the student
presses Analyse, never on an ordinary rerun, because each analysis is one
Gemini call. A stored analysis stays on screen after the CV, evidence or job
changes, with a note, because the student may be working through it. When
a different student is using the app, the stored analysis is cleared.
"""

from __future__ import annotations

import copy
import re

import streamlit as st

from app.cv_analyst import (
    ITEM_SECTIONS,
    SEVERITY_ORDER,
    analyse_cv,
    issue_entry,
)
from app.vector_store import load_jobs

ISSUES_KEY = "analysis_issues"
SNAPSHOT_KEY = "analysis_snapshot"
DIAGNOSTICS_KEY = "analysis_diagnostics"
ANALYSE_BUTTON_KEY = "analysis_run"
STALE_CAPTION = (
    "Your CV or evidence has changed since this analysis. Press Analyse again."
)
NOT_ANALYSED_CAPTION = "Press Analyse to check this CV against the selected job."
ALL_GEMINI_CAPTION = (
    "Explanations: written by Gemini; every one passed the grounding and "
    "wording checks."
)
FOOTER_CAPTION = (
    '"Not shown in your CV" describes the CV, not your ability, and nothing here '
    "predicts an employer's decision."
)
SEVERITY_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}
TYPE_LABELS = {
    "vague": "Vague",
    "no_evidence": "Not shown",
    "ungrounded_edit": "Not in your profile",
    "low_relevance": "Job match",
}
# Short, student-facing reasons for template text. Raw error details from
# diagnostics are never shown.
REASON_LABELS = {
    "disabled": "Gemini not used",
    "no_client": "no Gemini key configured",
    "api": "Gemini call failed",
    "parse": "Gemini reply was not valid JSON",
    "empty": "Gemini reply had no feedback",
    "private_section": "name and contact lines are not sent",
    "missing": "Gemini skipped the issue",
    "quote_mismatch": "Gemini text was about another line",
    "empty_field": "Gemini text was empty",
    "too_long": "Gemini text was too long",
    "name_placeholder": "Gemini text had a placeholder",
    "banned": "Gemini text broke a wording rule",
    "invented_number": "Gemini text added a number",
    "ungrounded": "Gemini text named something not in your profile",
}
_QUOTE_PREVIEW_CHARS = 60
_MARKDOWN_SPECIAL_RE = re.compile(r"([\\`*_\[\]<>#|~$])")


def render_analysis_panel() -> None:
    """Analyse button, then the stored analysis grouped by CV section."""
    st.session_state.setdefault(ISSUES_KEY, [])
    st.session_state.setdefault(SNAPSHOT_KEY, {})
    st.session_state.setdefault(DIAGNOSTICS_KEY, {})

    st.subheader("CV analysis")
    profile = st.session_state.get("profile")
    cv_sections = st.session_state.get("cv_sections") or {}
    selected_job = st.session_state.get("selected_job")
    if profile is None or not cv_sections or not selected_job:
        st.caption("Generate a CV for a job to analyse it.")
        return

    current = _snapshot(selected_job, cv_sections, profile)
    stored = st.session_state[SNAPSHOT_KEY]
    if stored and stored.get("student") != current["student"]:
        # A different student is using the app: the last one's issues must
        # not appear, even marked out of date.
        st.session_state[ISSUES_KEY] = []
        st.session_state[SNAPSHOT_KEY] = {}
        st.session_state[DIAGNOSTICS_KEY] = {}
    if st.button("Analyse", key=ANALYSE_BUTTON_KEY):
        job = _job_with_description(selected_job)
        diagnostics: dict = {}
        with st.spinner("Analysing your CV..."):
            issues = analyse_cv(profile, cv_sections, job, diagnostics=diagnostics)
        st.session_state[ISSUES_KEY] = [issue.model_dump() for issue in issues]
        st.session_state[SNAPSHOT_KEY] = current
        st.session_state[DIAGNOSTICS_KEY] = diagnostics

    snapshot = st.session_state[SNAPSHOT_KEY]
    if not snapshot:
        st.caption(NOT_ANALYSED_CAPTION)
        return
    if snapshot != current:
        st.caption(STALE_CAPTION)

    issues = st.session_state[ISSUES_KEY]
    if not issues:
        st.markdown("No issues found.")
        return
    counts = ", ".join(
        f"{sum(issue['severity'] == level for issue in issues)} {level}"
        for level in SEVERITY_ORDER
    )
    st.markdown(f"**{len(issues)} issues:** {counts}")
    caption = source_caption(st.session_state[DIAGNOSTICS_KEY])
    if caption:
        st.caption(caption)

    # Entries are looked up in the CV as it was analysed, so the groups stay
    # right while the student edits.
    for section, items in display_groups(issues, snapshot["cv_sections"]):
        st.markdown(f"#### {section}")
        for label, members in items:
            with st.expander(label):
                for issue in members:
                    _render_issue(issue, show_quote_heading=len(members) > 1)
    st.caption(FOOTER_CAPTION)


def display_groups(
    issues: list[dict], cv_sections: dict[str, str]
) -> list[tuple[str, list[tuple[str, list[dict]]]]]:
    """Issues as [(section, [(expander label, issues)])], most severe first.

    issues arrive most severe first, so sections appear in the order of their
    most severe issue, and "Job ad" is a section like any other. Low
    no_evidence issues of one project or role share an expander labelled
    "Team Course Planner: Flask, SQLite".
    """
    sections: dict[str, list[dict]] = {}
    for issue in issues:
        items = sections.setdefault(issue["section"], [])
        entry = ""
        if (
            issue["type"] == "no_evidence"
            and issue["severity"] == "low"
            and issue["section"] in ITEM_SECTIONS
        ):
            entry = issue_entry(issue["section"], issue["quote"], cv_sections)
        shared = next((item for item in items if entry and item["entry"] == entry), None)
        if shared is not None:
            shared["issues"].append(issue)
        else:
            items.append({"entry": entry, "issues": [issue]})

    groups = []
    for section, items in sections.items():
        labelled = []
        for item in items:
            first = item["issues"][0]
            head = f"{SEVERITY_LABELS[first['severity']]} · {TYPE_LABELS[first['type']]}"
            if item["entry"]:
                quotes = ", ".join(issue["quote"] for issue in item["issues"])
                labelled.append((f"{head} · {item['entry']}: {quotes}", item["issues"]))
            else:
                labelled.append((f"{head} · {_preview(first['quote'])}", item["issues"]))
        groups.append((section, labelled))
    return groups


def source_caption(diagnostics: dict) -> str:
    """One line on where the issue text came from, without error details."""
    entries = list((diagnostics.get("issues") or {}).values())
    if not entries:
        return ""
    from_gemini = sum(entry["source"] == "gemini" for entry in entries)
    if from_gemini == len(entries):
        return ALL_GEMINI_CAPTION
    reasons: dict[str, int] = {}
    for entry in entries:
        if entry["source"] == "template":
            label = REASON_LABELS.get(entry["reason"], "a check failed")
            reasons[label] = reasons.get(label, 0) + 1
    if len(reasons) == 1:
        listed = next(iter(reasons))
    else:
        listed = "; ".join(f"{label} ({count})" for label, count in reasons.items())
    if from_gemini == 0:
        return f"Explanations: deterministic templates ({listed})."
    return (
        f"Explanations: written by Gemini for {from_gemini} of {len(entries)} issues; "
        f"templates for the rest ({listed})."
    )


def _render_issue(issue: dict, show_quote_heading: bool) -> None:
    if show_quote_heading:
        st.markdown(f"**{_escape(issue['quote'])}**")
    else:
        st.markdown(f"> {_escape(issue['quote'])}")
    st.markdown(f"**Why:** {_escape(issue['why'])}")
    st.markdown(f"**Suggestion:** {_escape(issue['suggestion'])}")
    st.markdown(f"**Question:** {_escape(issue['question'])}")


def _snapshot(selected_job: dict, cv_sections: dict[str, str], profile) -> dict:
    """What an analysis depends on: the student, the job, the CV and the evidence.

    The student is the profile without its evidence, so adding evidence
    marks the analysis out of date instead of looking like a new student.
    """
    return {
        "student": profile.model_dump(exclude={"skill_evidence"}),
        "job_id": str(selected_job.get("document_id") or ""),
        "cv_sections": dict(cv_sections),
        "skill_evidence": copy.deepcopy(list(profile.skill_evidence)),
    }


def _job_with_description(selected_job: dict) -> dict:
    """The selected job with its description, looked up by id when missing.

    The job match dict in session state carries no description until the
    app stores one, so the built-in jobs are looked up in load_jobs().
    """
    job = dict(selected_job)
    if str(job.get("description") or "").strip():
        return job
    document_id = job.get("document_id")
    for record in load_jobs():
        if record.document_id == document_id:
            job["description"] = record.description
            job.setdefault("title", record.title)
            break
    return job


def _preview(quote: str) -> str:
    """The quote, cut at a word boundary to fit an expander label."""
    if len(quote) <= _QUOTE_PREVIEW_CHARS:
        return quote
    head = quote[: _QUOTE_PREVIEW_CHARS - 3]
    return (head.rsplit(" ", 1)[0] if " " in head else head).rstrip(" ,;:") + "..."


def _escape(text: str) -> str:
    """Show CV and Gemini text literally, not as markdown or LaTeX."""
    return _MARKDOWN_SPECIAL_RE.sub(r"\\\1", text)
