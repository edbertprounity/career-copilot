"""Job posting parser tests. No live Gemini calls."""

from __future__ import annotations

import struct
import zlib

import pymupdf
import pytest

import app.job_input as job_input
from app.job_input import (
    CUSTOM_JOB_ID_PREFIX,
    CUSTOM_JOB_SIMILARITY_SCORE,
    GEMINI_MODEL,
    JOB_PARSE_SYSTEM_INSTRUCTION,
    JOB_RESPONSE_SCHEMA,
    JobInputError,
    parse_job_description,
    parse_job_from_files,
    parsed_job_to_match,
)


SAMPLE_JOB_TEXT = (
    "Graduate AI Engineer at Example Labs, Sydney NSW. Full time. "
    "Requirements: Python, PyTorch. Nice to have: Docker. "
    "Build and evaluate machine-learning models."
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


def _pdf_bytes(page_count: int = 1, text: str = "Graduate AI Engineer Python") -> bytes:
    document = pymupdf.open()
    for _ in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _empty_text_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    return data


def _image_parts(parts: list) -> list:
    return [
        part
        for part in parts
        if getattr(part, "inline_data", None) is not None
        and part.inline_data.mime_type
        and part.inline_data.mime_type.startswith("image/")
    ]


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


def test_parse_job_description_returns_structured_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def stub(client, system_instruction, user_content, model=GEMINI_MODEL, response_schema=None):
        captured["client"] = client
        captured["instruction"] = system_instruction
        captured["text"] = user_content
        captured["model"] = model
        captured["schema"] = response_schema
        return dict(PARSED_PAYLOAD)

    monkeypatch.setattr(job_input, "call_gemini_json", stub)
    result = parse_job_description("fake-client", SAMPLE_JOB_TEXT)
    assert captured["client"] == "fake-client"
    assert captured["model"] == GEMINI_MODEL
    assert captured["text"] == SAMPLE_JOB_TEXT
    assert captured["instruction"] == JOB_PARSE_SYSTEM_INSTRUCTION
    assert "Do not invent" in captured["instruction"]
    assert captured["schema"] is JOB_RESPONSE_SCHEMA
    assert result["job_title"] == "Graduate AI Engineer"
    assert result["company"] == "Example Labs"
    assert result["required_skills"] == ["Python", "PyTorch"]
    assert result["preferred_skills"] == ["Docker"]
    assert "raw_summary" in result


def test_parse_job_description_normalises_blank_optionals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(*_args, **_kwargs):
        payload = dict(PARSED_PAYLOAD)
        payload["company"] = "n/a"
        payload["location"] = "null"
        payload["employment_type"] = "  "
        return payload

    monkeypatch.setattr(job_input, "call_gemini_json", stub)
    result = parse_job_description(object(), SAMPLE_JOB_TEXT)
    assert result["company"] is None
    assert result["location"] is None
    assert result["employment_type"] is None


def test_parse_job_description_rejects_empty_text() -> None:
    with pytest.raises(JobInputError, match="empty"):
        parse_job_description(object(), "  ")


def test_parse_job_description_rejects_url_only() -> None:
    with pytest.raises(JobInputError, match="fetching URLs is not supported"):
        parse_job_description(object(), "https://example.com/jobs/ai-engineer")


def test_parse_job_description_wraps_gemini_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(*_args, **_kwargs):
        raise job_input.GeminiAPIError("upstream")

    monkeypatch.setattr(job_input, "call_gemini_json", stub)
    with pytest.raises(JobInputError, match="job parse failed"):
        parse_job_description(object(), SAMPLE_JOB_TEXT)


def test_parse_job_description_wraps_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(*_args, **_kwargs):
        raise job_input.GeminiJSONParseError(
            "Gemini JSON response must be an object, got list"
        )

    monkeypatch.setattr(job_input, "call_gemini_json", stub)
    with pytest.raises(JobInputError, match="job parse failed"):
        parse_job_description(object(), SAMPLE_JOB_TEXT)


def test_parse_job_description_rejects_empty_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(*_args, **_kwargs):
        payload = dict(PARSED_PAYLOAD)
        payload["job_title"] = "   "
        return payload

    monkeypatch.setattr(job_input, "call_gemini_json", stub)
    with pytest.raises(JobInputError, match="job title is missing"):
        parse_job_description(object(), SAMPLE_JOB_TEXT)


def test_parsed_job_to_match_maps_fields_and_composes_description() -> None:
    match = parsed_job_to_match(PARSED_PAYLOAD)
    assert match["title"] == "Graduate AI Engineer"
    assert match["job_type"] == "Full time"
    assert match["location"] == "Sydney NSW"
    assert match["document_id"].startswith(CUSTOM_JOB_ID_PREFIX)
    assert len(match["document_id"]) == len(CUSTOM_JOB_ID_PREFIX) + 16
    assert match["similarity_score"] is CUSTOM_JOB_SIMILARITY_SCORE
    assert match["similarity_score"] is None
    assert match["gap_analysis"] is None
    description = match["description"]
    assert "Company: Example Labs" in description
    assert "Location: Sydney NSW" in description
    assert "Employment type: Full time" in description
    assert "Graduate AI Engineer role in Sydney" in description
    assert "- Build and evaluate machine-learning models" in description
    assert "Required skills: Python, PyTorch" in description
    assert "Preferred skills: Docker" in description
    assert "Domain keywords: Python, PyTorch, machine-learning" in description
    assert "Qualifications and education" not in description
    assert match["supporting_excerpt"] == PARSED_PAYLOAD["raw_summary"]


def test_parsed_job_to_match_document_id_is_deterministic() -> None:
    first = parsed_job_to_match(PARSED_PAYLOAD)
    second = parsed_job_to_match(dict(PARSED_PAYLOAD))
    assert first["document_id"] == second["document_id"]
    changed = dict(PARSED_PAYLOAD)
    changed["required_skills"] = ["Python", "PyTorch", "SQL"]
    assert parsed_job_to_match(changed)["document_id"] != first["document_id"]


def test_parsed_job_to_match_excerpt_skips_metadata_header() -> None:
    payload = dict(PARSED_PAYLOAD)
    payload["raw_summary"] = ""
    match = parsed_job_to_match(payload)
    assert match["supporting_excerpt"] != "Company: Example Labs"
    assert not match["supporting_excerpt"].startswith("Company:")
    assert not match["supporting_excerpt"].startswith("Location:")
    assert match["supporting_excerpt"] == "Build and evaluate machine-learning models"
    skills_only = dict(payload)
    skills_only["key_responsibilities"] = []
    skills_match = parsed_job_to_match(skills_only)
    assert skills_match["supporting_excerpt"] == "Required skills: Python, PyTorch"


def test_parsed_job_to_match_skips_missing_optional_sections() -> None:
    minimal = {
        "job_title": "Data Analyst",
        "company": None,
        "location": "  ",
        "employment_type": "n/a",
        "required_skills": ["SQL", "SQL", "  "],
        "preferred_skills": [],
        "key_responsibilities": [],
        "qualifications_and_education": ["  "],
        "domain_keywords": [],
        "raw_summary": "",
    }
    match = parsed_job_to_match(minimal)
    assert match["title"] == "Data Analyst"
    assert match["location"] is None
    assert match["job_type"] is None
    assert "Company:" not in match["description"]
    assert "Location:" not in match["description"]
    assert "Employment type:" not in match["description"]
    assert "Preferred skills:" not in match["description"]
    assert "Key responsibilities:" not in match["description"]
    assert "Qualifications and education:" not in match["description"]
    assert "Domain keywords:" not in match["description"]
    assert match["description"] == "Required skills: SQL"


def test_parsed_job_to_match_rejects_empty_title() -> None:
    payload = dict(PARSED_PAYLOAD)
    payload["job_title"] = "\n"
    with pytest.raises(JobInputError, match="job title is missing"):
        parsed_job_to_match(payload)


def test_parse_job_from_files_uses_image_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def stub(client, system_instruction, parts, response_schema=None, model=GEMINI_MODEL):
        captured["parts"] = parts
        captured["schema"] = response_schema
        captured["model"] = model
        return dict(PARSED_PAYLOAD)

    monkeypatch.setattr(job_input, "call_gemini_parts_json", stub)
    result = parse_job_from_files(
        object(), [(_png_bytes(), "ad.png", "image/png")]
    )
    assert captured["schema"] is JOB_RESPONSE_SCHEMA
    assert captured["model"] == GEMINI_MODEL
    images = _image_parts(captured["parts"])
    assert len(images) == 1
    assert images[0].inline_data.mime_type == "image/png"
    assert result["job_title"] == "Graduate AI Engineer"
    assert "conversion_errors" not in result


def test_parse_job_from_files_uses_pdf_text_not_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def stub(client, system_instruction, parts, response_schema=None, model=GEMINI_MODEL):
        captured["parts"] = parts
        captured["schema"] = response_schema
        return dict(PARSED_PAYLOAD)

    monkeypatch.setattr(job_input, "call_gemini_parts_json", stub)
    result = parse_job_from_files(
        object(),
        [(_pdf_bytes(1, "Graduate AI Engineer Python"), "ad.pdf", "application/pdf")],
    )
    assert captured["schema"] is JOB_RESPONSE_SCHEMA
    assert _image_parts(captured["parts"]) == []
    texts = [
        part.text or ""
        for part in captured["parts"]
        if getattr(part, "text", None)
    ]
    assert any("Graduate AI Engineer Python" in text for text in texts)
    assert any("extracted PDF text" in text for text in texts)
    assert result["job_title"] == "Graduate AI Engineer"


def test_parse_job_from_files_rejects_pdf_without_text_layer() -> None:
    with pytest.raises(JobInputError, match="no usable text layer"):
        parse_job_from_files(
            object(), [(_empty_text_pdf_bytes(), "scan.pdf", "application/pdf")]
        )


def test_parse_job_from_files_rejects_unreadable() -> None:
    with pytest.raises(JobInputError, match="not a readable PDF"):
        parse_job_from_files(object(), [(b"nope", "ad.pdf", "application/pdf")])


def test_parse_job_from_files_preserves_conversion_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(*_args, **_kwargs):
        return dict(PARSED_PAYLOAD)

    monkeypatch.setattr(job_input, "call_gemini_parts_json", stub)
    result = parse_job_from_files(
        object(),
        [
            (_png_bytes(), "ok.png", "image/png"),
            (b"nope", "bad.pdf", "application/pdf"),
        ],
    )
    assert result["job_title"] == "Graduate AI Engineer"
    assert any("not a readable PDF" in item for item in result["conversion_errors"])
