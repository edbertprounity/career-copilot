"""CV revision uses profile facts only. No Gemini."""

from app.cv_revise import revise_cv_sections
from app.schemas import ExperienceRecord, StudentProfile


def test_ungrounded_line_is_removed_and_vague_entry_uses_profile() -> None:
    profile = StudentProfile(
        name="Edber",
        experience=[
            ExperienceRecord(
                role="Volunteer tutor",
                description=(
                    "Ran math tutoring sessions for final-year students "
                    "preparing for their exam."
                ),
                type="Volunteer",
            )
        ],
    )
    sections = {
        "Experience": (
            "**Volunteer tutor**\n"
            "- Helped out.\n"
            "- Increased revenue by 400% with Kubernetes."
        )
    }
    revised, notes = revise_cv_sections(
        profile,
        sections,
        [
            {
                "type": "vague",
                "section": "Experience",
                "quote": "Helped out.",
            },
            {
                "type": "ungrounded_edit",
                "section": "Experience",
                "quote": "Increased revenue by 400% with Kubernetes.",
            },
            {
                "type": "low_relevance",
                "section": "Job ad",
                "quote": "machine-learning fundamentals",
            },
        ],
    )
    text = revised["Experience"]
    assert "Kubernetes" not in text
    assert "math tutoring sessions" in text
    assert any("Removed" in note for note in notes)
    assert any("Did not add" in note for note in notes)
