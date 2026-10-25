"""CV revision uses profile facts only. No Gemini."""

import json
from pathlib import Path

from app.cv_analyst import _sections_from_markdown, analyse_cv
from app.cv_generator import generate_cv_markdown
from app.cv_revise import revise_cv_sections
from app.schemas import ExperienceRecord, StudentProfile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RETAIL = (
    "Handled customer questions, worked with a small team during busy periods "
    "and followed written procedures."
)


def _maya_cv() -> tuple[StudentProfile, dict, dict[str, str]]:
    """Maya's confirmed profile, a job, and the CV sections the editor holds."""
    rows = (DATA_DIR / "student_profiles.jsonl").read_text(encoding="utf-8").splitlines()
    maya_row = next(row for row in rows if "student_profile_01" in row)
    maya = StudentProfile.model_validate_json(maya_row)
    maya.is_confirmed = True
    jobs = (DATA_DIR / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    job = json.loads(next(r for r in jobs if '"graduate_ai_engineer"' in r))
    sections = _sections_from_markdown(generate_cv_markdown(maya, job, job["description"]))
    return maya, job, sections


def _analyse_and_revise(profile, sections, job):
    """What the panel's Analyse and revise button does, without Gemini."""
    issues = [i.model_dump() for i in analyse_cv(profile, sections, job, use_gemini=False)]
    return revise_cv_sections(profile, dict(sections), issues)


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


def test_typed_skill_is_removed_from_its_list_line_only() -> None:
    maya, job, sections = _maya_cv()
    assert "**Languages:** Python, Java" in sections["Skills"]
    sections["Skills"] = sections["Skills"].replace(
        "**Languages:** Python, Java", "**Languages:** Python, Java, Kubernetes"
    )

    revised, notes = _analyse_and_revise(maya, sections, job)

    assert "**Languages:** Python, Java" in revised["Skills"].splitlines()
    assert "Kubernetes" not in revised["Skills"]
    other_lines = [l for l in sections["Skills"].splitlines() if "Languages" not in l]
    assert all(line in revised["Skills"].splitlines() for line in other_lines)
    removed = [note for note in notes if note.startswith("Removed")]
    assert removed == [
        "Removed “Kubernetes” from Skills because it is not in your profile."
    ]


def test_unsupported_sentence_is_removed_from_its_bullet_only() -> None:
    maya, job, sections = _maya_cv()
    assert f"- {RETAIL}" in sections["Experience"].splitlines()
    sections["Experience"] = sections["Experience"].replace(
        RETAIL, f"{RETAIL} Deployed the store app on Kubernetes."
    )

    revised, notes = _analyse_and_revise(maya, sections, job)

    assert f"- {RETAIL}" in revised["Experience"].splitlines()
    assert "Kubernetes" not in revised["Experience"]
    assert any("Deployed the store app on Kubernetes." in note for note in notes)


def test_technologies_line_keeps_its_label_and_other_items() -> None:
    sections = {
        "Projects": "**Planner**\n- Built it.\n- Technologies: Python, Flask, Terraform"
    }
    issue = {"type": "ungrounded_edit", "section": "Projects", "quote": "Terraform"}
    revised, _ = revise_cv_sections(StudentProfile(), sections, [issue])
    assert revised["Projects"] == "**Planner**\n- Built it.\n- Technologies: Python, Flask"


def test_list_line_is_dropped_only_when_no_items_are_left() -> None:
    sections = {"Skills": "**Languages:** Python\nKubernetes"}
    issue = {"type": "ungrounded_edit", "section": "Skills", "quote": "Kubernetes"}
    revised, _ = revise_cv_sections(StudentProfile(), sections, [issue])
    assert revised["Skills"] == "**Languages:** Python"


def test_quote_inside_a_longer_item_leaves_the_line_alone() -> None:
    sections = {"Skills": "**Frameworks & Tools:** Kubernetes Operators, Git"}
    issue = {"type": "ungrounded_edit", "section": "Skills", "quote": "Kubernetes"}
    revised, notes = revise_cv_sections(StudentProfile(), sections, [issue])
    assert revised["Skills"] == sections["Skills"]
    assert notes == []
