"""Chat attachment helpers. No live Gemini."""

from __future__ import annotations

from app.ui_chat import (
    classify_chat_intent,
    compose_attachment_note,
    split_chat_prompt,
)


def test_split_chat_prompt_accepts_plain_string() -> None:
    assert split_chat_prompt("  hello  ") == ("hello", [])
    assert split_chat_prompt(None) == ("", [])


def test_split_chat_prompt_reads_chat_input_value() -> None:
    class Prompt:
        text = "here is my transcript"
        files = [("bytes", "scan.png", "image/png")]

    text, files = split_chat_prompt(Prompt())
    assert text == "here is my transcript"
    assert files[0][1] == "scan.png"


def test_unconfirmed_attachment_is_evidence() -> None:
    files = [(b"x", "transcript.png", "image/png")]
    assert (
        classify_chat_intent(
            "this is my transcript",
            files,
            profile_confirmed=False,
            has_selected_job=False,
            chat_locked=False,
        )
        == "evidence"
    )


def test_confirmed_job_ad_text_is_job() -> None:
    text = (
        "About the role: build APIs.\n"
        "Key responsibilities: write Python.\n"
        "Requirements: SQL and dashboards.\n"
        "Qualifications: a computing degree.\n"
    ) * 3
    assert (
        classify_chat_intent(
            text,
            [],
            profile_confirmed=True,
            has_selected_job=False,
            chat_locked=True,
        )
        == "job"
    )


def test_locked_plain_text_stays_locked() -> None:
    assert (
        classify_chat_intent(
            "hello",
            [],
            profile_confirmed=True,
            has_selected_job=False,
            chat_locked=True,
        )
        == "locked"
    )


def test_compose_attachment_note_lists_extracted_facts_only() -> None:
    note = compose_attachment_note(
        "here you go",
        [(b"x", "transcript.pdf", "application/pdf")],
        {
            "skills": [{"skill": "Python"}],
            "education": [{"degree": "BSc Computing", "institution": "UTS"}],
            "work_experience": [],
            "projects": [],
            "certifications": [],
        },
    )
    assert "transcript.pdf" in note
    assert "skill: Python" in note
    assert "BSc Computing" in note
    assert "do not invent" in note
    assert "Student message: here you go" in note
