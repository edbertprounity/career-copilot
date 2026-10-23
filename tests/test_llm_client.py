"""Unit tests for the shared Gemini JSON helpers. No live API calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types

from app.llm_client import (
    GEMINI_MODEL,
    GeminiAPIError,
    GeminiJSONParseError,
    call_gemini_json,
    call_gemini_parts_json,
)


class _FakeModels:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.models = _FakeModels(response=response, error=error)


def _parts() -> list:
    return [
        types.Part.from_text(text="Identify the colour."),
        types.Part.from_bytes(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png"),
    ]


def test_call_gemini_parts_json_returns_object() -> None:
    client = _FakeClient(response=SimpleNamespace(text='{"colour": "red"}'))
    result = call_gemini_parts_json(
        client,
        "Return JSON only.",
        _parts(),
        response_schema={"type": "OBJECT", "properties": {"colour": {"type": "STRING"}}},
    )
    assert result == {"colour": "red"}


def test_call_gemini_parts_json_forwards_model_config_and_image_part() -> None:
    schema = {
        "type": "OBJECT",
        "properties": {"colour": {"type": "STRING"}},
        "required": ["colour"],
    }
    parts = _parts()
    client = _FakeClient(response=SimpleNamespace(text='{"colour": "red"}'))
    call_gemini_parts_json(
        client,
        "Return a JSON object only.",
        parts,
        response_schema=schema,
    )
    assert len(client.models.calls) == 1
    sent = client.models.calls[0]
    assert sent["model"] == GEMINI_MODEL
    assert sent["contents"] is parts
    assert len(sent["contents"]) == 2
    image_part = sent["contents"][1]
    assert image_part.inline_data is not None
    assert image_part.inline_data.mime_type == "image/png"
    config = sent["config"]
    assert config.system_instruction == "Return a JSON object only."
    assert config.response_mime_type == "application/json"
    assert config.response_schema == schema


def test_call_gemini_parts_json_uses_explicit_model() -> None:
    client = _FakeClient(response=SimpleNamespace(text='{"ok": true}'))
    call_gemini_parts_json(
        client,
        "Return JSON only.",
        _parts(),
        model="custom-model-name",
    )
    assert client.models.calls[0]["model"] == "custom-model-name"


@pytest.mark.parametrize("text", [None, "", "   "])
def test_call_gemini_parts_json_empty_response_raises(text: str | None) -> None:
    client = _FakeClient(response=SimpleNamespace(text=text))
    with pytest.raises(GeminiAPIError, match="empty response"):
        call_gemini_parts_json(client, "Return JSON only.", _parts())


def test_call_gemini_parts_json_malformed_json_raises() -> None:
    client = _FakeClient(response=SimpleNamespace(text="{not json"))
    with pytest.raises(GeminiJSONParseError, match="not valid JSON"):
        call_gemini_parts_json(client, "Return JSON only.", _parts())


@pytest.mark.parametrize(
    "text, kind",
    [
        ('["red"]', "list"),
        ('"red"', "str"),
        ("12", "int"),
        ("true", "bool"),
        ("null", "NoneType"),
    ],
)
def test_call_gemini_parts_json_non_object_json_raises(text: str, kind: str) -> None:
    client = _FakeClient(response=SimpleNamespace(text=text))
    with pytest.raises(GeminiJSONParseError, match=kind):
        call_gemini_parts_json(client, "Return JSON only.", _parts())


def test_call_gemini_parts_json_sdk_exception_becomes_api_error() -> None:
    client = _FakeClient(error=RuntimeError("upstream unavailable"))
    with pytest.raises(GeminiAPIError, match="generate_content failed") as caught:
        call_gemini_parts_json(client, "Return JSON only.", _parts())
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_call_gemini_parts_json_reraises_existing_api_error() -> None:
    original = GeminiAPIError("already wrapped")
    client = _FakeClient(error=original)
    with pytest.raises(GeminiAPIError, match="already wrapped") as caught:
        call_gemini_parts_json(client, "Return JSON only.", _parts())
    assert caught.value is original


def test_call_gemini_json_still_sends_text_contents() -> None:
    client = _FakeClient(response=SimpleNamespace(text='{"status": "ok"}'))
    result = call_gemini_json(
        client,
        "Reply with a JSON object only.",
        'Return {"status": "ok"} as JSON',
        response_schema={"type": "OBJECT"},
    )
    assert result == {"status": "ok"}
    sent = client.models.calls[0]
    assert sent["model"] == GEMINI_MODEL
    assert sent["contents"] == 'Return {"status": "ok"} as JSON'
    assert sent["config"].response_schema == {"type": "OBJECT"}
