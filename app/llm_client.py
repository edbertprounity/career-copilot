"""Shared Gemini client for Student Career Copilot."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

_client: genai.Client | None = None


class GeminiAPIError(Exception):
    """Raised when a Gemini API call fails or returns an empty response."""


class GeminiJSONParseError(Exception):
    """Raised when a Gemini response is not valid JSON or is not an object."""


def get_gemini_client() -> genai.Client | None:
    """Return a cached Gemini client, or None if the key is missing or init fails."""
    global _client
    if _client is not None:
        return _client

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        return None

    try:
        _client = genai.Client(api_key=api_key)
    except Exception:
        return None

    return _client


def call_gemini_json(
    client: genai.Client,
    system_instruction: str,
    user_content: str,
    model: str = "gemini-3.6-flash",
    response_schema: dict | None = None,
) -> dict:
    """Call Gemini and return a parsed JSON object.

    Raises:
        GeminiAPIError: if the API call fails or the response is empty.
        GeminiJSONParseError: if the response is not a JSON object.
    """
    config_kwargs: dict = {
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
    }
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except GeminiAPIError:
        raise
    except Exception as exc:
        raise GeminiAPIError(f"Gemini generate_content failed: {exc}") from exc

    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise GeminiAPIError("Gemini returned an empty response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiJSONParseError(
            f"Gemini response was not valid JSON: {exc.msg}"
        ) from exc

    if not isinstance(parsed, dict):
        raise GeminiJSONParseError(
            f"Gemini JSON response must be an object, got {type(parsed).__name__}"
        )

    return parsed


if __name__ == "__main__":
    load_dotenv()
    test_client = get_gemini_client()
    if test_client is None:
        print(
            "ERROR: Could not initialize Gemini client. "
            "Check that GEMINI_API_KEY is set in .env."
        )
        raise SystemExit(1)

    try:
        result = call_gemini_json(
            test_client,
            system_instruction="Reply with a JSON object only. No markdown.",
            user_content='Return {"status": "ok"} as JSON',
        )
        print("SUCCESS:", result)
    except (GeminiAPIError, GeminiJSONParseError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"ERROR: Unexpected failure: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
