"""Shared Gemini client for Student Career Copilot."""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

_client: genai.Client | None = None

GEMINI_MODEL = "gemini-3.6-flash"
UNAVAILABLE_RETRY_SECONDS = 2


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


def _json_generate_config(
    system_instruction: str,
    response_schema: dict | None,
) -> types.GenerateContentConfig:
    config_kwargs: dict = {
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
    }
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema
    return types.GenerateContentConfig(**config_kwargs)


def _parse_json_object_response(response: object) -> dict:
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


def _is_unavailable(exc: BaseException) -> bool:
    """True for a busy-model 503 UNAVAILABLE from the Gemini SDK."""
    if getattr(exc, "code", None) == 503:
        return True
    text = str(exc)
    return "503" in text and "UNAVAILABLE" in text


def _generate_content(
    client: genai.Client,
    *,
    model: str,
    contents: object,
    config: types.GenerateContentConfig,
) -> object:
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )


def _generate_json_object(
    client: genai.Client,
    *,
    model: str,
    contents: object,
    system_instruction: str,
    response_schema: dict | None,
) -> dict:
    config = _json_generate_config(system_instruction, response_schema)
    try:
        try:
            response = _generate_content(
                client, model=model, contents=contents, config=config
            )
        except GeminiAPIError:
            raise
        except Exception as exc:
            if not _is_unavailable(exc):
                raise GeminiAPIError(
                    f"Gemini generate_content failed: {exc}"
                ) from exc
            time.sleep(UNAVAILABLE_RETRY_SECONDS)
            try:
                response = _generate_content(
                    client, model=model, contents=contents, config=config
                )
            except GeminiAPIError:
                raise
            except Exception as retry_exc:
                raise GeminiAPIError(
                    f"Gemini generate_content failed: {retry_exc}"
                ) from retry_exc
    except GeminiAPIError:
        raise
    except Exception as exc:
        raise GeminiAPIError(f"Gemini generate_content failed: {exc}") from exc

    return _parse_json_object_response(response)


def call_gemini_json(
    client: genai.Client,
    system_instruction: str,
    user_content: str,
    model: str = GEMINI_MODEL,
    response_schema: dict | None = None,
) -> dict:
    """Call Gemini and return a parsed JSON object.

    Raises:
        GeminiAPIError: if the API call fails or the response is empty.
        GeminiJSONParseError: if the response is not a JSON object.
    """
    return _generate_json_object(
        client,
        model=model,
        contents=user_content,
        system_instruction=system_instruction,
        response_schema=response_schema,
    )


def call_gemini_parts_json(
    client: genai.Client,
    system_instruction: str,
    parts: list,
    response_schema: dict | None = None,
    model: str = GEMINI_MODEL,
) -> dict:
    """Call Gemini with SDK Part objects and return a parsed JSON object.

    ``parts`` is a list of ``google.genai.types.Part`` (text and/or inline
    bytes such as ``image/png``). It is passed as ``contents``; this helper
    does not upload files or create a second client.

    Raises:
        GeminiAPIError: if the API call fails or the response is empty.
        GeminiJSONParseError: if the response is not a JSON object.
    """
    return _generate_json_object(
        client,
        model=model,
        contents=parts,
        system_instruction=system_instruction,
        response_schema=response_schema,
    )


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
