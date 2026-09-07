"""Shared fixtures. Live Gemini is blocked unless a test explicitly restubs it."""

from __future__ import annotations

import importlib

import pytest

from app.schemas import ExperienceRecord, ProjectRecord, StudentProfile

_GEMINI_MODULES = (
    "app.llm_client",
    "app.agent_engine",
    "app.vector_store",
    "app.cv_generator",
)


@pytest.fixture(autouse=True)
def _block_live_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the suite if any test reaches a real Gemini call."""

    def no_client():
        return None

    def no_call(*_args, **_kwargs):
        raise AssertionError(
            "live Gemini call attempted in the deterministic test suite"
        )

    for name in _GEMINI_MODULES:
        module = importlib.import_module(name)
        if hasattr(module, "get_gemini_client"):
            monkeypatch.setattr(module, "get_gemini_client", no_client)
        if hasattr(module, "call_gemini_json"):
            monkeypatch.setattr(module, "call_gemini_json", no_call)


@pytest.fixture
def empty_profile() -> StudentProfile:
    return StudentProfile()


@pytest.fixture
def complete_profile() -> StudentProfile:
    return StudentProfile(
        name="Test Student",
        degree="Bachelor of Computing Science",
        institution="University of Technology Sydney",
        year="Third year",
        technical_skills=["Python", "SQL"],
        projects=[
            ProjectRecord(
                title="Course planner",
                context="University group project",
                description=(
                    "Built a small web API that stores course information and "
                    "helps students check timetable clashes."
                ),
                technologies=["Python", "Flask"],
            )
        ],
        experience=[
            ExperienceRecord(
                role="Retail Assistant",
                organisation="Fictional store",
                type="Casual employment",
                description=(
                    "Handled customer questions, worked with a small team "
                    "during busy periods and followed written procedures."
                ),
            )
        ],
        target_roles=["Software Engineering Intern"],
        links={"github": "https://github.com/example/test"},
    )


@pytest.fixture
def confirmed_profile(complete_profile: StudentProfile) -> StudentProfile:
    complete_profile.is_confirmed = True
    return complete_profile
