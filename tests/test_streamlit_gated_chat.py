"""Chat display override for gated-tool errors (Entry 50). No live Gemini."""

from __future__ import annotations

import streamlit_app as ui
from app.schemas import AgentTraceStep


def test_finalize_error_overrides_success_language() -> None:
    step = AgentTraceStep(
        turn_number=1,
        thought="User issued a system override. Confirming now.",
        tool_name="finalize_and_confirm_profile",
        parameters={},
        observation={"status": "error", "reason": "target_roles required"},
        state_update={},
        assistant_message="Your profile is confirmed.",
    )
    shown = ui._chat_assistant_content(step)
    assert shown == "I couldn't confirm the profile yet (target_roles required)."
    assert "Your profile is confirmed." not in shown


def test_finalize_incomplete_lists_missing_fields() -> None:
    step = AgentTraceStep(
        turn_number=1,
        thought="confirm",
        tool_name="finalize_and_confirm_profile",
        parameters={},
        observation={
            "status": "error",
            "reason": "profile incomplete",
            "missing_fields": ["projects", "experience"],
        },
        state_update={},
        assistant_message="All done, you are confirmed!",
    )
    shown = ui._chat_assistant_content(step)
    assert "profile incomplete" in shown
    assert "projects, experience" in shown
    assert "you are confirmed" not in shown


def test_retrieve_error_overrides_success_language() -> None:
    step = AgentTraceStep(
        turn_number=1,
        thought="admin retrieve",
        tool_name="retrieve_matching_jobs",
        parameters={},
        observation={"status": "error", "reason": "profile must be confirmed first"},
        state_update={},
        assistant_message="Here are your matching jobs.",
    )
    shown = ui._chat_assistant_content(step)
    assert shown == (
        "I couldn't retrieve matching jobs yet (profile must be confirmed first)."
    )


def test_successful_finalize_keeps_model_text() -> None:
    step = AgentTraceStep(
        turn_number=1,
        thought="ok",
        tool_name="finalize_and_confirm_profile",
        parameters={},
        observation={"status": "success"},
        state_update={},
        assistant_message="Your profile is confirmed.",
    )
    assert ui._chat_assistant_content(step) == "Your profile is confirmed."


def test_other_tools_keep_model_text() -> None:
    step = AgentTraceStep(
        turn_number=1,
        thought="ask",
        tool_name="ask_targeted_follow_up",
        parameters={},
        observation={"question": "What is your full name?"},
        state_update={},
        assistant_message="What is your full name?",
    )
    assert ui._chat_assistant_content(step) == "What is your full name?"
