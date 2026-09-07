"""Streamlit frontend for Student Career Copilot."""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import streamlit as st

import app.vector_store as vector_store
from app.agent_engine import inspect_completeness, run_agent_turn
import app.llm_client as llm_client
from app.cv_generator import (
    cv_layout,
    generate_cv_markdown,
    generate_cv_pdf,
    generate_cv_summary,
)
from app.schemas import StudentProfile
from app.vector_store import load_jobs, search_top_jobs

SECTION_ORDER = (
    "Summary",
    "Contact",
    "Education",
    "Skills",
    "Professional Summary",
    "Projects",
    "Experience",
    "Activities",
)
HEADING_TO_DISPLAY = {
    "Contact": "Contact",
    "Education": "Education",
    "Skills": "Skills",
    "Professional Summary": "Professional Summary",
    "Projects": "Projects",
    "Experience": "Experience",
    "Activities": "Activities",
}
DISPLAY_TO_HEADING = {display: heading for heading, display in HEADING_TO_DISPLAY.items()}
FIXED_BODY_SECTIONS = ("Contact", "Education")
REORDERABLE_SECTIONS = (
    "Skills",
    "Professional Summary",
    "Projects",
    "Experience",
    "Activities",
)
SUMMARY_FALLBACK_REASONS = {
    "no_client": "no Gemini key configured",
    "api": "Gemini call failed",
    "empty": "Gemini returned no summary",
    "ungrounded": "grounding check rejected the Gemini text",
}
# Python-gated tools. If the observation is an error, chat must show that
# reason rather than a possibly-false assistant_message (Entry 49).
GATED_CHAT_TOOLS = frozenset(
    {"finalize_and_confirm_profile", "retrieve_matching_jobs"}
)
WIDGET_PREFIX = "cvedit_"
ORDER_PREFIX = "cvorder_"
DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILES_PATH = DATA_DIR / "student_profiles.jsonl"
INTROS_PATH = DATA_DIR / "student_introductions.jsonl"
CLEAR_AFTER_CONFIRMATION = "clear after confirmation"
SKIP_NOTE = (
    "Profile already complete — skipping straight to job matches."
)
BLANK_ACTION = "blank"


def _widget_key(section_name: str) -> str:
    return f"{WIDGET_PREFIX}{section_name}"


def _init_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = StudentProfile()
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "turn_number" not in st.session_state:
        st.session_state.turn_number = 0
    if "job_matches" not in st.session_state:
        st.session_state.job_matches = []
    if "selected_job" not in st.session_state:
        st.session_state.selected_job = None
    if "cv_sections" not in st.session_state:
        st.session_state.cv_sections = {}
    if "cv_sections_original" not in st.session_state:
        st.session_state.cv_sections_original = {}
    if "cv_section_order" not in st.session_state:
        st.session_state.cv_section_order = []
    if "cv_pdf_sections" not in st.session_state:
        st.session_state.cv_pdf_sections = {}
    if "cv_pdf_order" not in st.session_state:
        st.session_state.cv_pdf_order = []
    if "cv_summary_path" not in st.session_state:
        st.session_state.cv_summary_path = {}
    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = []
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "pending_reset_section" not in st.session_state:
        st.session_state.pending_reset_section = None
    if "pending_demo_action" not in st.session_state:
        st.session_state.pending_demo_action = None
    if "skip_to_jobs_note" not in st.session_state:
        st.session_state.skip_to_jobs_note = False


def _clear_cv_widget_keys() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(WIDGET_PREFIX) or str(key).startswith(ORDER_PREFIX):
            del st.session_state[key]


def _reset_session() -> None:
    _clear_cv_widget_keys()
    st.session_state.profile = StudentProfile()
    st.session_state.conversation_history = []
    st.session_state.turn_number = 0
    st.session_state.job_matches = []
    st.session_state.selected_job = None
    st.session_state.cv_sections = {}
    st.session_state.cv_sections_original = {}
    st.session_state.cv_section_order = []
    st.session_state.cv_pdf_sections = {}
    st.session_state.cv_pdf_order = []
    st.session_state.cv_summary_path = {}
    st.session_state.agent_trace = []
    st.session_state.pdf_bytes = None
    st.session_state.pending_reset_section = None
    st.session_state.pending_demo_action = None
    st.session_state.skip_to_jobs_note = False


def _apply_pending_section_reset() -> None:
    name = st.session_state.pending_reset_section
    if not name:
        return
    original = st.session_state.cv_sections_original.get(name, "")
    st.session_state[_widget_key(name)] = original
    st.session_state.cv_sections[name] = original
    st.session_state.pending_reset_section = None


@lru_cache(maxsize=1)
def _demo_catalog() -> tuple[dict, ...]:
    intros: dict[str, dict] = {}
    with INTROS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                intros[row["document_id"]] = row
    profiles: dict[str, dict] = {}
    with PROFILES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                profiles[row["document_id"]] = row
    cards = []
    for document_id in sorted(intros):
        intro = intros[document_id]
        profile_row = profiles[document_id]
        cards.append(
            {
                "document_id": document_id,
                "name": profile_row.get("name") or document_id,
                "profile_status": intro.get("profile_status") or "",
                "introduction": intro.get("initial_introduction") or "",
                "profile_row": profile_row,
            }
        )
    return tuple(cards)


def _onboarding_message() -> str:
    missing = inspect_completeness(StudentProfile())["missing_fields"]
    groups: list[str] = []
    if "name" in missing:
        groups.append("your name")
    if any(field in missing for field in ("degree", "institution", "year")):
        groups.append("your education")
    if "technical_skills" in missing:
        groups.append("technical skills")
    if "projects" in missing:
        groups.append("projects")
    if "experience" in missing:
        groups.append("work experience")
    if "target_roles" in missing:
        groups.append("the roles you're targeting")
    if not groups:
        listed = "your background"
    elif len(groups) == 1:
        listed = groups[0]
    elif len(groups) == 2:
        listed = f"{groups[0]} and {groups[1]}"
    else:
        listed = ", ".join(groups[:-1]) + f", and {groups[-1]}"
    return (
        f"Tell me about {listed} — I'll ask follow-up questions for anything unclear."
    )


def _onboarding_visible() -> bool:
    return (
        not st.session_state.conversation_history
        and not st.session_state.job_matches
    )


@st.cache_resource
def _cached_retrieval_bundle():
    load_jobs()
    return vector_store._model, vector_store._jobs, vector_store._embeddings


def _ensure_retrieval_ready() -> None:
    model, jobs, embeddings = _cached_retrieval_bundle()
    vector_store._model = model
    vector_store._jobs = jobs
    vector_store._embeddings = embeddings


def _search_top_jobs_without_gemini(profile: StudentProfile) -> list[dict]:
    _ensure_retrieval_ready()
    original = vector_store.get_gemini_client
    vector_store.get_gemini_client = lambda: None
    try:
        return search_top_jobs(profile)
    finally:
        vector_store.get_gemini_client = original


def _load_clear_demo_profile(card: dict) -> None:
    _reset_session()
    profile = StudentProfile.model_validate(card["profile_row"])
    profile.is_confirmed = True
    st.session_state.profile = profile
    st.session_state.job_matches = _search_top_jobs_without_gemini(profile)
    st.session_state.skip_to_jobs_note = True


def _start_follow_up_demo(card: dict) -> None:
    _reset_session()
    _handle_user_message(card["introduction"])


def _apply_pending_demo_action() -> None:
    action = st.session_state.pending_demo_action
    if not action:
        return
    st.session_state.pending_demo_action = None
    if action == BLANK_ACTION:
        _reset_session()
        return
    card = next((item for item in _demo_catalog() if item["document_id"] == action), None)
    if card is None:
        return
    if card["profile_status"] == CLEAR_AFTER_CONFIRMATION:
        _load_clear_demo_profile(card)
        return
    _start_follow_up_demo(card)


def _split_cv_markdown(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = "Summary"
    buffer: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            text = "\n".join(buffer).strip()
            if text:
                sections[current] = text
            heading = line[3:].strip()
            current = HEADING_TO_DISPLAY.get(heading, heading)
            buffer = []
        else:
            buffer.append(line)
    text = "\n".join(buffer).strip()
    if text:
        sections[current] = text
    return {name: body for name, body in sections.items() if body.strip()}


def _cv_editor_order() -> list[str]:
    stored = [name for name in (st.session_state.get("cv_section_order") or []) if name != "Summary"]
    if not stored:
        stored = [name for name in SECTION_ORDER if name != "Summary"]
    return ["Summary", *stored]


def _order_widget_key(section_name: str) -> str:
    return f"{ORDER_PREFIX}{section_name}"


def _position_label(position: int) -> str:
    if 10 <= position % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th")
    return f"{position}{suffix}"


def _fixed_body_sections() -> list[str]:
    present = st.session_state.cv_sections
    return [name for name in FIXED_BODY_SECTIONS if name in present]


def _present_reorderable() -> list[str]:
    present = {
        name
        for name in REORDERABLE_SECTIONS
        if name in st.session_state.cv_sections
    }
    current = [
        name
        for name in (st.session_state.get("cv_section_order") or [])
        if name in present
    ]
    leftover = [name for name in REORDERABLE_SECTIONS if name in present and name not in current]
    return current + leftover


def _sync_order_widget_keys(
    movable: list[str], skip: str | None = None
) -> None:
    for index, name in enumerate(movable, start=1):
        if name == skip:
            continue
        st.session_state[_order_widget_key(name)] = index


def _apply_reorder_position(section: str, position: int) -> None:
    movable = _present_reorderable()
    if section not in movable:
        return
    movable.remove(section)
    index = max(0, min(position, len(movable) + 1) - 1)
    movable.insert(index, section)
    st.session_state.cv_section_order = _fixed_body_sections() + movable
    _sync_order_widget_keys(movable, skip=section)


def _on_reorder_select(section: str) -> None:
    if cv_layout(st.session_state.profile) == "two-column":
        return
    position = int(st.session_state[_order_widget_key(section)])
    _apply_reorder_position(section, position)


def _reassemble_cv(sections: dict[str, str]) -> str:
    parts: list[str] = []
    summary = sections.get("Summary", "").strip()
    if summary:
        parts.append(summary)
    for display in _cv_editor_order():
        if display == "Summary":
            continue
        body = sections.get(display, "").strip()
        if not body:
            continue
        heading = DISPLAY_TO_HEADING.get(display, display)
        parts.append(f"## {heading}\n{body}")
    return "\n\n".join(parts) + "\n"


def _job_description(document_id: str) -> str:
    for job in load_jobs():
        if job.document_id == document_id:
            return job.description
    return ""


def _chat_locked() -> bool:
    if st.session_state.job_matches:
        return True
    if not st.session_state.agent_trace:
        return False
    return st.session_state.agent_trace[-1].get("requires_input") is False


def _last_tool_name() -> str:
    if not st.session_state.agent_trace:
        return ""
    return str(st.session_state.agent_trace[-1].get("tool_name") or "")


def _maybe_load_jobs() -> None:
    profile = st.session_state.profile
    if not profile.is_confirmed or st.session_state.job_matches:
        return

    if st.session_state.agent_trace:
        step = st.session_state.agent_trace[-1]
        if step.get("tool_name") == "retrieve_matching_jobs":
            matches = (step.get("observation") or {}).get("matches")
            if matches:
                st.session_state.job_matches = matches
                return

    st.session_state.job_matches = search_top_jobs(profile)


def _handle_user_message(user_message: str) -> None:
    profile = st.session_state.profile
    st.session_state.turn_number += 1
    step = run_agent_turn(
        profile,
        st.session_state.conversation_history,
        user_message,
        st.session_state.turn_number,
    )
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_message}
    )
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": _chat_assistant_content(step)}
    )
    st.session_state.agent_trace.append(step.model_dump())
    st.session_state.profile = profile
    _maybe_load_jobs()


def _chat_assistant_content(step) -> str:
    """Message shown in chat for one turn.

    For finalize_and_confirm_profile and retrieve_matching_jobs, Python is the
    source of truth. If observation.status is "error", show that reason even
    when assistant_message claims success. Other tools keep the model text.
    """
    observation = step.observation if isinstance(getattr(step, "observation", None), dict) else {}
    tool_name = str(getattr(step, "tool_name", "") or "")
    if tool_name in GATED_CHAT_TOOLS and observation.get("status") == "error":
        return _gated_tool_error_message(tool_name, observation)
    return str(getattr(step, "assistant_message", "") or "")


def _gated_tool_error_message(tool_name: str, observation: dict) -> str:
    reason = str(observation.get("reason") or "").strip() or "that action was refused"
    if tool_name == "retrieve_matching_jobs":
        return f"I couldn't retrieve matching jobs yet ({reason})."
    missing = observation.get("missing_fields")
    text = f"I couldn't confirm the profile yet ({reason})."
    if isinstance(missing, list) and missing:
        listed = ", ".join(str(item) for item in missing if str(item).strip())
        if listed:
            text += f" Still missing: {listed}."
    return text


def _select_job(job: dict) -> None:
    profile = st.session_state.profile
    description = _job_description(str(job.get("document_id") or ""))
    title = str(job.get("title") or "")
    client = llm_client.get_gemini_client()
    diagnostics: dict = {}
    if client is None:
        diagnostics = {"path": "fallback", "reason": "no_client", "detail": ""}
        summary = None
    else:
        summary = generate_cv_summary(
            client, profile, description, title, diagnostics=diagnostics
        )
    st.session_state.cv_summary_path = diagnostics
    markdown = generate_cv_markdown(
        profile, job, description, professional_summary=summary
    )
    sections = _split_cv_markdown(markdown)
    _clear_cv_widget_keys()
    st.session_state.selected_job = job
    st.session_state.cv_sections = sections
    st.session_state.cv_sections_original = dict(sections)
    st.session_state.cv_section_order = [name for name in sections if name != "Summary"]
    for name, content in sections.items():
        st.session_state[_widget_key(name)] = content
    _sync_order_widget_keys(_present_reorderable())
    _store_cv_pdf(markdown)


def _sync_sections_from_widgets() -> None:
    for name in list(st.session_state.cv_sections):
        key = _widget_key(name)
        if key in st.session_state:
            st.session_state.cv_sections[name] = st.session_state[key]


def _render_sidebar_reset() -> None:
    st.sidebar.header("Session")
    if st.sidebar.button("Start New Profile"):
        _reset_session()
        st.rerun()


def _render_sidebar_status_and_trace() -> None:
    completeness = inspect_completeness(st.session_state.profile)
    st.sidebar.header("Profile status")
    if st.session_state.profile.is_confirmed:
        st.sidebar.success("Profile confirmed")
    else:
        st.sidebar.info("Profile not confirmed")

    if completeness["complete"]:
        st.sidebar.write("Required fields: complete")
    else:
        missing = ", ".join(completeness["missing_fields"]) or "none"
        st.sidebar.write(f"Missing: {missing}")

    st.sidebar.header("Agent Trace")
    if not st.session_state.agent_trace:
        st.sidebar.caption("No turns yet.")
        return
    for step in st.session_state.agent_trace:
        tool = step.get("tool_name") or "none"
        follow = (step.get("observation") or {}).get("follow_on") or {}
        follow_tool = follow.get("tool_name")
        label = f"Turn {step.get('turn_number')} — {tool}"
        if follow_tool and follow_tool != tool:
            label += f" → {follow_tool}"
        with st.sidebar.expander(label):
            st.markdown(f"**thought:** {step.get('thought') or ''}")
            st.markdown(f"**tool_name:** `{step.get('tool_name') or ''}`")
            st.write("observation")
            st.json(step.get("observation") or {})
            st.write("state_update")
            st.json(step.get("state_update") or {})


@st.fragment
def _render_chat_fragment() -> None:
    _render_sidebar_status_and_trace()
    st.subheader("Conversation")
    if _onboarding_visible():
        with st.chat_message("assistant"):
            st.write(_onboarding_message())
    if st.session_state.skip_to_jobs_note:
        st.success(SKIP_NOTE)
    for item in st.session_state.conversation_history:
        role = item.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        with st.chat_message(role):
            st.write(item.get("content") or "")

    if _chat_locked():
        if st.session_state.job_matches:
            st.info(
                "Chat is paused because matching jobs are loaded. "
                "This avoids a second retrieval. Use Start New Profile to begin again."
            )
        elif _last_tool_name() == "step_limit_reached":
            st.info(
                "We've reached the maximum number of turns for this session. "
                "The last message above lists what's still missing. "
                "Use Start New Profile in the sidebar to begin again."
            )
        else:
            st.info("The agent is no longer accepting input for this profile.")
        return

    prompt = st.chat_input("Describe your background, or answer the agent's question.")
    if prompt and prompt.strip():
        had_jobs = bool(st.session_state.job_matches)
        with st.spinner("Thinking..."):
            _handle_user_message(prompt.strip())
        if st.session_state.job_matches and not had_jobs:
            st.rerun()
        st.rerun(scope="fragment")


def _render_demo_cards() -> None:
    if not _onboarding_visible():
        return
    st.subheader("Try a demo profile")
    cards = list(_demo_catalog())
    cards.append(
        {
            "document_id": BLANK_ACTION,
            "name": "Start with my own profile",
            "profile_status": "blank session — no pre-loaded data",
        }
    )
    for row_start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, card in zip(columns, cards[row_start : row_start + 3], strict=False):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{card['name']}**")
                    st.caption(card["profile_status"])
                    if st.button("Use this profile", key=f"demo_{card['document_id']}"):
                        st.session_state.pending_demo_action = card["document_id"]
                        st.rerun()


def _render_jobs() -> None:
    if not st.session_state.job_matches:
        return

    st.subheader("Matching jobs")
    selected_id = None
    if st.session_state.selected_job:
        selected_id = st.session_state.selected_job.get("document_id")

    for job in st.session_state.job_matches:
        document_id = job.get("document_id")
        title = job.get("title") or document_id or "Untitled job"
        score = job.get("similarity_score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        with st.container(border=True):
            heading = title
            if document_id == selected_id:
                heading = f"{title} (selected)"
            st.markdown(f"**{heading}**")
            st.caption(f"similarity_score: {score_text}")
            st.write(job.get("supporting_excerpt") or "")
            st.write(job.get("gap_analysis") or "")
            if st.button("Generate CV for this job", key=f"select_{document_id}"):
                _select_job(job)
                st.rerun()


def _store_cv_pdf(markdown: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        output_path = handle.name
    try:
        generate_cv_pdf(markdown, output_path)
        st.session_state.pdf_bytes = Path(output_path).read_bytes()
    finally:
        Path(output_path).unlink(missing_ok=True)
    st.session_state.cv_pdf_sections = dict(st.session_state.cv_sections)
    st.session_state.cv_pdf_order = list(st.session_state.cv_section_order)


def _cv_pdf_is_stale() -> bool:
    current = st.session_state.cv_sections
    snapshot = st.session_state.get("cv_pdf_sections") or {}
    if set(current) != set(snapshot):
        return True
    if any(current.get(name, "") != snapshot.get(name, "") for name in current):
        return True
    current_order = [
        name
        for name in (st.session_state.get("cv_section_order") or [])
        if name in current
    ]
    pdf_order = [
        name
        for name in (st.session_state.get("cv_pdf_order") or [])
        if name in current
    ]
    return current_order != pdf_order


def _render_section_order_control() -> None:
    movable = _present_reorderable()
    if not movable:
        return
    two_column = cv_layout(st.session_state.profile) == "two-column"
    st.markdown("**Section order**")
    if two_column:
        st.caption("Section order is fixed in two-column layout")
    positions = list(range(1, len(movable) + 1))
    for name in movable:
        key = _order_widget_key(name)
        if key not in st.session_state:
            st.session_state[key] = movable.index(name) + 1
        st.selectbox(
            name,
            options=positions,
            format_func=_position_label,
            key=key,
            disabled=two_column,
            on_change=None if two_column else _on_reorder_select,
            args=() if two_column else (name,),
        )


def _summary_path_caption() -> str:
    """One line stating whether the summary came from Gemini or the template."""
    info = st.session_state.get("cv_summary_path") or {}
    path = str(info.get("path") or "")
    if not path:
        return ""
    if path == "gemini":
        return "Professional Summary: Gemini-generated, grounding check passed."
    reason = SUMMARY_FALLBACK_REASONS.get(
        str(info.get("reason") or ""), "Gemini not used"
    )
    detail = " ".join(str(info.get("detail") or "").split())
    if detail:
        reason = f"{reason} — {detail[:140]}"
    return f"Professional Summary: deterministic template ({reason})."


def _render_cv_editor() -> None:
    if not st.session_state.selected_job or not st.session_state.cv_sections:
        return

    st.subheader("CV editor")
    caption = _summary_path_caption()
    if caption:
        st.caption(caption)
    _sync_sections_from_widgets()
    _render_section_order_control()
    stale = _cv_pdf_is_stale()
    rerun_after_widgets = False
    col_actions, _ = st.columns([1, 2])
    with col_actions:
        if stale:
            st.caption(
                "The CV has changed — regenerate the PDF to update the download."
            )
            if st.button("Regenerate PDF"):
                markdown = _reassemble_cv(st.session_state.cv_sections)
                _store_cv_pdf(markdown)
                rerun_after_widgets = True

    if st.session_state.pdf_bytes:
        st.download_button(
            "Download CV PDF",
            data=st.session_state.pdf_bytes,
            file_name="career_copilot_cv.pdf",
            mime="application/pdf",
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Edit sections**")
        for name in _cv_editor_order():
            if name not in st.session_state.cv_sections:
                continue
            key = _widget_key(name)
            current = st.session_state.get(key, st.session_state.cv_sections[name])
            original = st.session_state.cv_sections_original.get(name, "")
            label = f"{name} (edited)" if current != original else name
            height = 180 if name in {"Summary", "Professional Summary", "Projects", "Experience"} else 120
            st.text_area(label, key=key, height=height)
            st.session_state.cv_sections[name] = st.session_state[key]
            if st.button(f"Reset {name}", key=f"reset_{name}"):
                st.session_state.pending_reset_section = name
                st.rerun()

    with right:
        st.markdown("**Live preview**")
        st.markdown(_reassemble_cv(st.session_state.cv_sections))

    if rerun_after_widgets:
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Student Career Copilot", layout="wide")
    _init_state()
    _apply_pending_section_reset()
    st.title("Student Career Copilot")
    with st.spinner("Loading local job index..."):
        _ensure_retrieval_ready()
    _apply_pending_demo_action()
    _render_sidebar_reset()
    _render_demo_cards()
    _render_chat_fragment()
    _render_jobs()
    _render_cv_editor()


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    get_script_run_ctx = None

if get_script_run_ctx is not None and get_script_run_ctx() is not None:
    main()
elif __name__ == "__main__":
    main()
