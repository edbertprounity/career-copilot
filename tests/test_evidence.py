"""Evidence extraction pipeline tests. No live Gemini calls."""

from __future__ import annotations

import struct
import zlib

import pymupdf
import pytest

import app.evidence as ev
from app.evidence import (
    EXTRACT_SYSTEM_INSTRUCTION,
    MAX_FILE_BYTES,
    MAX_PDF_PAGES,
    EvidenceError,
    extract_evidence_from_files,
    extract_pdf_text,
    files_to_parts,
    resolve_mime_type,
)


def _png_bytes(width: int = 8, height: int = 8, rgb: bytes = b"\xff\x00\x00") -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + rgb * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _pdf_bytes(page_count: int, text: str = "Python Flask") -> bytes:
    document = pymupdf.open()
    for _ in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), text)
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


def test_resolve_mime_from_extension_and_alias() -> None:
    assert resolve_mime_type("shot.PNG", "") == "image/png"
    assert resolve_mime_type("a.jpg", "jpg") == "image/jpeg"
    assert resolve_mime_type("a.jpeg", "image/jpg") == "image/jpeg"
    assert resolve_mime_type("doc.pdf", "application/pdf") == "application/pdf"
    assert resolve_mime_type("x.webp", ".webp") == "image/webp"


def test_resolve_mime_rejects_unknown() -> None:
    with pytest.raises(EvidenceError, match="unsupported"):
        resolve_mime_type("notes.docx", "application/vnd.openxmlformats")


def test_png_becomes_image_part() -> None:
    parts, errors = files_to_parts([(_png_bytes(), "card.png", "image/png")])
    assert errors == []
    images = _image_parts(parts)
    assert len(images) == 1
    assert images[0].inline_data.mime_type == "image/png"


def test_jpeg_magic_bytes_required() -> None:
    parts, errors = files_to_parts([(b"not-a-jpeg", "x.jpg", "image/jpeg")])
    assert parts == []
    assert any("not a readable JPEG" in item for item in errors)


def test_empty_file_is_error() -> None:
    parts, errors = files_to_parts([(b"", "blank.png", "image/png")])
    assert parts == []
    assert any("empty" in item for item in errors)


def test_oversized_file_is_error() -> None:
    blob = b"\x89PNG\r\n\x1a\n" + b"x" * (MAX_FILE_BYTES + 1)
    parts, errors = files_to_parts([(blob, "huge.png", "image/png")])
    assert parts == []
    assert any("exceeds" in item for item in errors)


def test_pdf_pages_become_png_parts() -> None:
    parts, errors = files_to_parts([(_pdf_bytes(2), "cv.pdf", "application/pdf")])
    assert errors == []
    images = _image_parts(parts)
    assert len(images) == 2
    assert all(part.inline_data.mime_type == "image/png" for part in images)
    labels = [
        part.text
        for part in parts
        if getattr(part, "text", None)
    ]
    assert any(text and "page 1" in text for text in labels)
    assert any(text and "page 2" in text for text in labels)


def test_pdf_page_limit() -> None:
    parts, errors = files_to_parts(
        [(_pdf_bytes(MAX_PDF_PAGES + 2), "long.pdf", "pdf")]
    )
    assert errors == []
    assert len(_image_parts(parts)) == MAX_PDF_PAGES
    intro = " ".join(part.text or "" for part in parts if getattr(part, "text", None))
    assert "extra pages omitted" in intro


def test_corrupt_pdf_is_error() -> None:
    parts, errors = files_to_parts([(b"this is not a pdf", "bad.pdf", "application/pdf")])
    assert parts == []
    assert any("not a readable PDF" in item for item in errors)


def test_extract_pdf_text_returns_embedded_text() -> None:
    text = extract_pdf_text(
        _pdf_bytes(1, "Graduate AI Engineer Python Flask"), "ad.pdf"
    )
    assert "Graduate AI Engineer Python Flask" in text


def test_extract_pdf_text_rejects_empty_text_layer() -> None:
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    with pytest.raises(EvidenceError, match="no usable text layer"):
        extract_pdf_text(data, "scan.pdf")


def test_extract_pdf_text_rejects_corrupt_pdf() -> None:
    with pytest.raises(EvidenceError, match="not a readable PDF"):
        extract_pdf_text(b"this is not a pdf", "bad.pdf")


def test_extract_skips_gemini_when_no_readable_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("live Gemini call attempted")

    monkeypatch.setattr(ev, "call_gemini_parts_json", boom)
    result = extract_evidence_from_files(
        object(), [(b"nope", "x.docx", "application/msword")]
    )
    assert result["skills"] == []
    assert result["errors"]


def test_extract_evidence_from_files_uses_schema_and_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def stub(client, system_instruction, parts, response_schema=None, model=None):
        captured["instruction"] = system_instruction
        captured["parts"] = parts
        captured["schema"] = response_schema
        captured["client"] = client
        return {
            "work_experience": [],
            "education": [],
            "skills": [
                {
                    "skill": "Python",
                    "filename": "card.png",
                    "page": None,
                    "excerpt": "Python",
                    "visibility": "explicit",
                }
            ],
            "projects": [],
            "certifications": [],
            "uncertain_findings": [],
            "errors": [],
        }

    monkeypatch.setattr(ev, "call_gemini_parts_json", stub)
    png = _png_bytes()
    result = extract_evidence_from_files(
        "fake-client", [(png, "card.png", "image/png")]
    )
    assert captured["client"] == "fake-client"
    assert captured["instruction"] == EXTRACT_SYSTEM_INSTRUCTION
    assert "Do not invent" in captured["instruction"]
    assert captured["schema"] is ev.EXTRACT_RESPONSE_SCHEMA
    images = _image_parts(captured["parts"])
    assert len(images) == 1
    assert images[0].inline_data.mime_type == "image/png"
    assert result["skills"][0]["skill"] == "Python"
    assert result["errors"] == []


def test_extract_keeps_file_errors_with_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stub(*_args, **_kwargs):
        return {
            "work_experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": [],
            "uncertain_findings": [],
            "errors": [],
        }

    monkeypatch.setattr(ev, "call_gemini_parts_json", stub)
    result = extract_evidence_from_files(
        object(),
        [
            (_png_bytes(), "ok.png", "image/png"),
            (b"broken", "no.pdf", "application/pdf"),
        ],
    )
    assert any("not a readable PDF" in item for item in result["errors"])
