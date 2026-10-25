"""Apply analyst findings to the CV using only facts already on the profile.

Does not call Gemini and does not invent employers, skills, or results.
"""

from __future__ import annotations

import re

from app.schemas import StudentProfile

_ITEM_SECTIONS = ("Projects", "Experience")
_SKILLS_SECTION = "Skills"
# Prefixes kept when an item or sentence is removed from a line.
_TECH_PREFIX_RE = re.compile(r"^(\s*-\s*technologies\s*:\s*)(.*)$", re.IGNORECASE)
_SKILLS_PREFIX_RE = re.compile(r"^(\s*(?:[-*•]\s+)?(?:\*\*[^*]+:\*\*\s*)?)(.*)$")
_PROSE_PREFIX_RE = re.compile(r"^(\s*(?:#+\s*|[-*•]\s+)?)(.*)$")


def revise_cv_sections(
    profile: StudentProfile,
    cv_sections: dict[str, str],
    issues: list[dict],
) -> tuple[dict[str, str], list[str]]:
    """Return updated section text and student-facing notes about what changed."""
    sections = {name: text for name, text in cv_sections.items()}
    notes: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        kind = str(issue.get("type") or "")
        section = str(issue.get("section") or "")
        quote = str(issue.get("quote") or "").strip()
        if kind == "ungrounded_edit" and quote and section in sections:
            updated = _remove_quote(sections[section], quote, section)
            if updated != sections[section]:
                sections[section] = updated
                notes.append(f"Removed “{quote}” from {section} because it is not in your profile.")
        elif kind == "vague" and section in _ITEM_SECTIONS:
            updated, note = _expand_entry(profile, section, sections.get(section, ""), quote)
            if note:
                sections[section] = updated
                notes.append(note)
        elif kind == "no_evidence" and quote:
            updated, note = _show_profile_use(profile, sections, section, quote)
            if note and updated:
                name, text = updated
                sections[name] = text
                notes.append(note)
        elif kind == "low_relevance" and quote:
            note = _relevance_note(profile, quote)
            if note:
                notes.append(note)
    return sections, notes


def _remove_quote(text: str, quote: str, section: str) -> str:
    """Remove the quoted claim, not the line around it.

    A list line (Skills, or "- Technologies:") loses only the item equal to
    the quote, keeping its **Label:** and the other items. A prose line loses
    only the quoted sentence. A line is dropped only when nothing is left.
    """
    kept: list[str] = []
    for line in text.splitlines():
        if quote not in line:
            kept.append(line)
            continue
        tech = _TECH_PREFIX_RE.match(line)
        if tech or section == _SKILLS_SECTION:
            prefix, body = (tech or _SKILLS_PREFIX_RE.match(line)).groups()
            items = [item.strip() for item in body.split(",") if item.strip()]
            remaining = [item for item in items if item != quote]
            if remaining == items:
                kept.append(line)  # the quote is inside an item, not one: leave it
            elif remaining:
                kept.append(prefix + ", ".join(remaining))
            continue
        prefix, body = _PROSE_PREFIX_RE.match(line).groups()
        rest = " ".join(body.replace(quote, " ").split())
        if rest:
            kept.append(prefix + rest)
    return "\n".join(kept).strip()


def _profile_entries(profile: StudentProfile, section: str) -> list[tuple[str, str]]:
    if section == "Projects":
        return [
            (item.title.strip(), item.description.strip())
            for item in profile.projects
            if item.title.strip() and item.description.strip()
        ]
    return [
        (item.role.strip(), item.description.strip())
        for item in profile.experience
        if item.role.strip() and item.description.strip()
    ]


def _expand_entry(
    profile: StudentProfile, section: str, text: str, quote: str
) -> tuple[str, str]:
    del quote
    for title, description in _profile_entries(profile, section):
        if title.lower() not in text.lower():
            continue
        if description.lower() in text.lower():
            return text, ""
        return (
            f"{text.rstrip()}\n- {description}",
            f"Expanded {title} using the description already on your profile.",
        )
    return text, ""


def _blob(profile: StudentProfile) -> str:
    parts = [
        profile.degree,
        profile.institution,
        " ".join(profile.technical_skills),
        " ".join(profile.activities),
        " ".join(profile.target_roles),
    ]
    parts.extend(item.description for item in profile.projects)
    parts.extend(item.description for item in profile.experience)
    parts.extend(
        str(item.get("source") or "")
        for item in profile.skill_evidence
        if isinstance(item, dict)
    )
    return "\n".join(part for part in parts if part)


def _show_profile_use(
    profile: StudentProfile,
    sections: dict[str, str],
    section: str,
    skill: str,
) -> tuple[tuple[str, str] | None, str]:
    """Put a profile sentence that already names the skill into the CV."""
    needle = skill.lower()
    if needle in "\n".join(sections.values()).lower():
        return None, ""
    sentence = ""
    for _title, description in (
        *_profile_entries(profile, "Projects"),
        *_profile_entries(profile, "Experience"),
    ):
        if needle in description.lower():
            sentence = description
            break
    if not sentence:
        return None, f"Left {skill} unchanged — your profile does not say where you used it."
    target = section if section in _ITEM_SECTIONS and section in sections else "Experience"
    if target not in sections:
        target = next((name for name in _ITEM_SECTIONS if name in sections), "")
    if not target:
        return None, ""
    return (
        (target, f"{sections[target].rstrip()}\n- {sentence}"),
        f"Added your own description of {skill} to {target}.",
    )


def _relevance_note(profile: StudentProfile, requirement: str) -> str:
    if requirement.lower() in _blob(profile).lower():
        return ""
    return (
        f"Did not add “{requirement}” — it is not in your profile, "
        "so the CV will not claim it."
    )
