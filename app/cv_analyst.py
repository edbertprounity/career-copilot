"""CV analyst: finds weak points in the CV the student is editing.

Python decides every issue. The confirmed profile is the ground truth, and
cv_sections, the editor's current text keyed by section name, is what gets
checked, so the student's own edits are analysed too.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from app.agent_engine import THIN_DESCRIPTION_WORDS
from app.schemas import StudentProfile

IssueType = Literal["vague", "no_evidence", "ungrounded_edit", "low_relevance"]
Severity = Literal["high", "medium", "low"]

SEVERITY_ORDER = ("high", "medium", "low")
SKILLS_SECTION = "Skills"
# Sections the generator writes as entries: a **bold** header line, then "- "
# description bullets and an optional "- Technologies:" line.
ITEM_SECTIONS = ("Projects", "Experience")

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
# One marker then a space, or a bare marker. "**Languages:**" is not a bullet.
_BULLET_RE = re.compile(r"^[-*•](?:\s+|$)")
_TECH_LINE_RE = re.compile(r"^-\s*technologies\s*:\s*", re.IGNORECASE)
_SKILL_LABEL_RE = re.compile(r"^\*\*[^*]+:\*\*\s*")

# vague signals. Each is a fixed rule, so a flag can always be explained.
# Tool: two or more capitals (API, CI, PyTorch) or a + / # (C++, C#).
_TECH_SHAPE_RE = re.compile(
    r"\b[A-Za-z]*[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*\b|[A-Za-z][+#]"
)
# Single-capital tools the shape rule misses. Case-sensitive whole words, so
# "excel at" is not a tool.
_COMMON_TOOLS = (
    "Excel", "PowerPoint", "Outlook", "Microsoft", "Slack", "Jira", "Trello",
    "Figma", "Canva", "Tableau", "Salesforce", "Zendesk", "Shopify", "Xero",
    "Linux", "Docker", "Python", "Java",
)
_COMMON_TOOL_RE = re.compile(r"\b(?:" + "|".join(_COMMON_TOOLS) + r")\b")
# Acronyms that name a team, role or place, not a tool. The institution's and
# the entry's organisation's initials are added per entry.
_NON_TOOL_ACRONYMS = frozenset({"IT", "HR", "CEO", "NSW"})
_OUTCOME_RE = re.compile(
    r"%"
    r"|\b(?:improv|increas|reduc|decreas|enabl|achiev|award|deliver|launch|"
    r"deploy|adopt|resolv)\w*"
    r"|\b(?:saved|cut|faster|accuracy|precision|recall|latency|won|"
    r"resulted in|led to|so that|used by)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "dozen",
    "hundred", "hundreds", "thousand", "thousands", "million",
)
_SCALE_RE = re.compile(
    r"\d|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE
)


class AnalysisIssue(BaseModel):
    issue_id: str
    section: str
    quote: str
    type: IssueType
    severity: Severity
    why: str
    suggestion: str
    question: str


@dataclass
class _CvItem:
    """One Projects or Experience entry as it currently reads in the editor."""

    title: str
    organisation: str = ""
    bullets: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        return " ".join(self.bullets)


def analyse_cv(
    profile: StudentProfile, cv_sections: dict[str, str], job: dict | None
) -> list[AnalysisIssue]:
    """Return the CV's issues, most severe first, then in CV section order.

    profile is the confirmed ground truth, cv_sections the editor's text by
    section name, and job the selected job (title and description) or None.
    """
    issues = _vague_issues(cv_sections, profile)
    _make_ids_unique(issues)
    section_rank = {name: index for index, name in enumerate(cv_sections)}
    return sorted(
        issues,
        key=lambda issue: (
            SEVERITY_ORDER.index(issue.severity),
            section_rank.get(issue.section, len(section_rank)),
        ),
    )


def _vague_issues(
    cv_sections: dict[str, str], profile: StudentProfile
) -> list[AnalysisIssue]:
    tools = [*_skills_items(cv_sections), *_profile_technologies(profile)]
    institution_acronyms = _name_acronyms(profile.institution)
    issues: list[AnalysisIssue] = []
    for section in ITEM_SECTIONS:
        for item in _parse_items(cv_sections.get(section, "")):
            not_tools = (
                _NON_TOOL_ACRONYMS
                | institution_acronyms
                | _name_acronyms(item.organisation)
            )
            issue = _vague_issue(section, item, tools, not_tools)
            if issue is not None:
                issues.append(issue)
    return issues


def _vague_issue(
    section: str, item: _CvItem, tools: list[str], not_tools: frozenset[str]
) -> AnalysisIssue | None:
    """Flag an entry that is too short, or that has no tool, outcome or scale.

    Too short uses detect_thin_content's threshold, applied to the CV text so
    edits count. Otherwise all three signals must be missing.
    """
    quote = item.bullets[0] if item.bullets else item.title
    if not quote:
        return None
    name = _entry_name(section, item.title)
    words = len(item.description.split())
    if words < THIN_DESCRIPTION_WORDS:
        if words == 0:
            why = "This entry has no description, so it doesn't show what you did."
        else:
            plural = "" if words == 1 else "s"
            why = (
                f"This description is only {words} word{plural}, "
                "too short to show what you did."
            )
        return _issue(
            "vague",
            section,
            quote,
            "high",
            why=why,
            suggestion=(
                "Add what you built or did, the tool you used, and one result "
                "or number, for example how many users, records or teammates."
            ),
            question=f"What did you do in {name}, what did you use, and what came of it?",
        )
    if (
        _has_tool(item, tools, not_tools)
        or _OUTCOME_RE.search(item.description)
        or _SCALE_RE.search(item.description)
    ):
        return None
    return _issue(
        "vague",
        section,
        quote,
        "medium",
        why=(
            "This says what you did, but not which tool you used, how big the "
            "work was, or what came of it."
        ),
        suggestion=(
            "Add one detail you can back up: a tool, a number (people, "
            "requests, records), or a result."
        ),
        question=(
            f"In {name}, what did you use, how much did you handle, and what "
            "changed because of your work?"
        ),
    )


def _has_tool(item: _CvItem, tools: list[str], not_tools: frozenset[str]) -> bool:
    """True for any tool signal in the entry.

    A Technologies line, a technology the CV or profile names, a common tool,
    or a tech-shaped token that is not a team, role or place acronym.
    """
    description = item.description
    if item.technologies:
        return True
    if any(_contains_phrase(description, tool) for tool in tools):
        return True
    if _COMMON_TOOL_RE.search(description):
        return True
    return any(
        match.group(0) not in not_tools
        for match in _TECH_SHAPE_RE.finditer(description)
    )


def _profile_technologies(profile: StudentProfile) -> list[str]:
    """Every technology the profile names: skills, project tech, evidence."""
    return [
        *profile.technical_skills,
        *(tech for project in profile.projects for tech in project.technologies),
        *(
            str(item.get("skill") or "")
            for item in profile.skill_evidence
            if isinstance(item, dict)
        ),
    ]


def _name_acronyms(name: str) -> frozenset[str]:
    """A name's initials and any acronyms inside it.

    UTS for University of Technology Sydney, so a mention of the place is not
    read as a tool.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", name)
    found = {word for word in words if _TECH_SHAPE_RE.fullmatch(word)}
    initials = "".join(word[0] for word in words if word[0].isupper())
    if len(initials) >= 2:
        found.add(initials)
    return frozenset(found)


def _entry_name(section: str, title: str) -> str:
    noun = "project" if section == "Projects" else "role"
    return f"the {title} {noun}" if title else f"this {noun}"


def _issue(
    issue_type: IssueType,
    section: str,
    quote: str,
    severity: Severity,
    *,
    why: str,
    suggestion: str,
    question: str,
) -> AnalysisIssue:
    return AnalysisIssue(
        issue_id=_issue_id(issue_type, section, quote),
        section=section,
        quote=quote,
        type=issue_type,
        severity=severity,
        why=why,
        suggestion=suggestion,
        question=question,
    )


def _issue_id(issue_type: str, section: str, quote: str) -> str:
    """Stable across re-runs, so evidence can be linked to an issue."""
    digest = hashlib.sha1(f"{issue_type}|{section}|{quote}".encode("utf-8"))
    return f"{issue_type}-{digest.hexdigest()[:8]}"


def _make_ids_unique(issues: list[AnalysisIssue]) -> None:
    """Suffix repeats (two identical entries) in detection order: -2, -3."""
    seen: dict[str, int] = {}
    for issue in issues:
        count = seen.get(issue.issue_id, 0) + 1
        seen[issue.issue_id] = count
        if count > 1:
            issue.issue_id = f"{issue.issue_id}-{count}"


def _parse_items(body: str) -> list[_CvItem]:
    """Split a Projects/Experience section into entries.

    Edited text is read leniently: an unbulleted line counts as description,
    and text before the first header becomes an untitled entry. Bullet text is
    kept verbatim so it can be quoted back.
    """
    items: list[_CvItem] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("**"):
            title, organisation = _header_parts(line)
            items.append(_CvItem(title=title, organisation=organisation))
            continue
        if not items:
            items.append(_CvItem(title=""))
        current = items[-1]
        tech_label = _TECH_LINE_RE.match(line)
        if tech_label:
            current.technologies.extend(_split_list(line[tech_label.end() :]))
            continue
        text = _BULLET_RE.sub("", line)
        if text:
            current.bullets.append(text)
    return items


def _header_parts(line: str) -> tuple[str, str]:
    """Entry name and organisation from a header line.

    Project headers are "**title** (context)". Experience headers are
    "**role, organisation, type**", so the first comma part names the entry
    and the second is the organisation.
    """
    match = _BOLD_RE.search(line)
    text = match.group(1) if match else line.strip("* ")
    parts = [part.strip() for part in text.split(",")]
    return parts[0], parts[1] if len(parts) > 1 else ""


def _skills_items(cv_sections: dict[str, str]) -> list[str]:
    """Every item the Skills section lists, without the **Category:** labels."""
    items: list[str] = []
    for raw_line in cv_sections.get(SKILLS_SECTION, "").splitlines():
        line = _SKILL_LABEL_RE.sub("", _BULLET_RE.sub("", raw_line.strip()))
        items.extend(_split_list(line))
    return items


def _split_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def _contains_phrase(text: str, phrase: str) -> bool:
    """Whole-phrase match, so SQL does not match inside SQLite."""
    key = _normalize(phrase)
    if not key:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])"
    return re.search(pattern, _normalize(text)) is not None
