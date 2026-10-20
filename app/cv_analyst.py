"""CV analyst: finds weak points in the CV the student is editing.

Python decides every issue. The confirmed profile is the ground truth, and
cv_sections, the editor's current text keyed by section name, is what gets
checked, so the student's own edits are analysed too. Gemini only rewrites
each issue's why, suggestion and question, and every rewritten field is
checked; a field that fails keeps the issue's template text.

Standalone run, no Streamlit, deterministic unless --gemini is given:
    python -m app.cv_analyst <profile_id> <job_id> [--json] [--gemini]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

# Called through the module (llm_client.call_gemini_json), so the test
# suite's stub on app.llm_client also covers this module.
from app import llm_client
from app.agent_engine import THIN_DESCRIPTION_WORDS, update_profile_field
from app.cv_generator import (
    LAYOUT_COMMENT_RE,
    generate_cv_markdown,
    profile_supports,
    ungrounded_terms,
)
from app.schemas import StudentProfile
from app.vector_store import JOBS_PATH

IssueType = Literal["vague", "no_evidence", "ungrounded_edit", "low_relevance"]
Severity = Literal["high", "medium", "low"]

SEVERITY_ORDER = ("high", "medium", "low")
SKILLS_SECTION = "Skills"
# Sections the generator writes as entries: a **bold** header line, then "- "
# description bullets and an optional "- Technologies:" line.
ITEM_SECTIONS = ("Projects", "Experience")
# Section of low_relevance issues about job requirements the CV never
# addresses. Their quote comes from the job ad, not the CV.
JOB_AD_SECTION = "Job ad"
MAX_REQUIREMENT_ISSUES = 5
PROFILES_PATH = JOBS_PATH.parent / "student_profiles.jsonl"

# Gemini feedback. The template text doubles as the draft Gemini rewrites, so
# the draft carries every fact the issue rests on.
FEEDBACK_FIELDS = ("why", "suggestion", "question")
MAX_FIELD_CHARS = 300
UNAVAILABLE_RETRY_SECONDS = 2
# The name line and contact details never leave the machine.
_PRIVATE_SECTIONS = frozenset({"Summary", "Contact"})
_FEEDBACK_SYSTEM_INSTRUCTION = (
    "You write feedback for problems that a Python checker has already found in "
    "a student's CV. You never find new problems, and you never change an "
    "issue's type, severity or quote. Each issue comes with its quote (the CV "
    "text it is about, or for section 'Job ad' the job ad's words), context (the "
    "CV section it sits in, possibly empty) and a draft why, suggestion and "
    "question that state its facts. Rewrite the draft to be specific and "
    "encouraging. why: one sentence on why this weakens the CV for a reader. "
    "suggestion: one concrete edit, using only facts in the quote, context and "
    "draft; where a fact is missing, write a bracketed placeholder such as "
    "[number of users] instead of inventing one. question: one question that "
    "asks the student for the missing fact. Rules: never say or imply that the "
    "student lacks a skill or experience; say it is not shown in the CV. Never "
    "predict hiring, shortlisting, or what an employer will think or do. Do not "
    "name any technology, employer, number, award or result that is not in the "
    "quote, context or draft. For ungrounded_edit, suggest removing the text or "
    "backing it up with evidence, never expanding it. Keep each field under 40 "
    "words. Return every issue exactly once with its issue_id and its quote "
    "copied exactly. Reply with JSON only."
)
_FEEDBACK_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "issues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "issue_id": {"type": "STRING"},
                    "quote": {"type": "STRING"},
                    "why": {"type": "STRING"},
                    "suggestion": {"type": "STRING"},
                    "question": {"type": "STRING"},
                },
                "required": ["issue_id", "quote", "why", "suggestion", "question"],
            },
        },
    },
    "required": ["issues"],
}
# Rule 3 of the brief, enforced on Gemini text: no claim that the student lacks
# something and no prediction about hiring.
_BANNED_WORDING_RE = re.compile(
    r"\b(?:lack\w*|(?:do not|don't|dont) have|not qualified|unqualified|"
    r"hir(?:e|ed|es|ing)|employers? will|shortlist\w*|reject\w*|"
    r"your chances|not a fit)\b",
    re.IGNORECASE,
)
# Redaction sends [name]; Gemini may echo it, so kept text gets the first name.
_NAME_PLACEHOLDER_RE = re.compile(r"\[name\]", re.IGNORECASE)
# Rule 2 for numbers: a figure in digits must come from the issue itself.
# Bracketed placeholders such as [number of users] are removed first.
_DIGIT_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^#+\s*")
# Same split as the summary grounding check: claims cannot cross sentences.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
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

# no_evidence: achievement claims that a certificate or record could back.
_ACHIEVEMENT_RE = re.compile(
    r"\b(?:award\w*|prize\w*|won|winner\w*|scholarship\w*|certif\w*|"
    r"dean's list|first place|top \d+%)",
    re.IGNORECASE,
)
# Stemming for clearing no_evidence. Both sides get the same treatment, so
# over-stemming only makes clearing likelier, which is the safer error.
_STEM_SUFFIXES = ("ing", "ed", "s")
_MIN_STEM = 3
_TOKEN_EDGE_PUNCTUATION = ".,;:()[]!?\"'"

# low_relevance: requirement blocks in a job ad. "Desirable experience
# includes" opens a block mid-sentence, without a colon.
_JOB_HEADER_RE = re.compile(
    r"\b(?:preferred experience includes|desirable experience includes|"
    r"(?:requirements|minimum qualifications|preferred qualifications|"
    r"nice to have|preferred|desirable|responsibilities|about the role)\s*:)",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r";|\n|(?<=\.)\s+")
# Eligibility, not skills: evidence cannot address it, and judging it would
# edge into predicting hiring.
_ELIGIBILITY_RE = re.compile(
    r"\b(?:degree|bachelor\w*|master\w*|phd|doctorate|postgraduate|years?|"
    r"qualifications?)\b",
    re.IGNORECASE,
)
_LEAD_IN_RE = re.compile(
    r"^(?:and|or|a|an|the|strong|solid|firm|good|basic|working|proven|"
    r"knowledge of|understanding of|experience (?:with|in)|familiarity with|"
    r"proficiency (?:in|with)|ability to|at least one of|preferably in)\s+",
    re.IGNORECASE,
)
# Words that say how much, not what, so they never count as addressing a
# requirement: "Experience with Docker" needs Docker, not the word experience.
_GENERIC_REQUIREMENT_WORDS = frozenset(
    {
        # The review's list.
        "experience", "knowledge", "understanding", "skills", "skill",
        "ability", "strong", "working", "familiarity", "proficiency", "good",
        "basic", "solid",
        # Grammar and degree words.
        "and", "or", "the", "of", "in", "with", "for", "to", "on", "at", "by",
        "as", "from", "into", "such", "including", "other", "any", "using",
        "least", "one", "more", "preferably", "related", "similar", "modern",
        "equivalent", "field", "fields",
    }
)
_WORD_RE = re.compile(r"[A-Za-z0-9+#]+")
# Items that only point at something else ("or similar frameworks") name no
# requirement of their own.
_FILLER_ITEM_RE = re.compile(
    r"^(?:similar|other|related|equivalent|comparable)\b", re.IGNORECASE
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
    profile: StudentProfile,
    cv_sections: dict[str, str],
    job: dict | None,
    *,
    use_gemini: bool = True,
    diagnostics: dict | None = None,
) -> list[AnalysisIssue]:
    """Return the CV's issues, most severe first, then in CV section order.

    profile is the confirmed ground truth, cv_sections the editor's text by
    section name, and job the selected job (title and description) or None.
    The generator's layout comment is stripped first, so no check sees it.

    With use_gemini and a client available, one batched Gemini call rewrites
    the issues' text. diagnostics, when given, is filled with where each
    issue's text came from and why, so a caller can show it.
    """
    sections = {
        name: LAYOUT_COMMENT_RE.sub("", text) for name, text in cv_sections.items()
    }
    job_title = str((job or {}).get("title") or "")
    issues = [
        *_vague_issues(sections, profile),
        *_ungrounded_edit_issues(sections, profile, job_title),
        *_no_evidence_issues(sections, profile, job_title),
        *_low_relevance_issues(sections, profile, job),
    ]
    _make_ids_unique(issues)
    section_rank = {name: index for index, name in enumerate(cv_sections)}
    issues.sort(
        key=lambda issue: (
            SEVERITY_ORDER.index(issue.severity),
            section_rank.get(issue.section, len(section_rank)),
        ),
    )
    report = _write_feedback(issues, sections, profile, job_title, use_gemini)
    if diagnostics is not None:
        diagnostics.update(report)
    return issues


def add_skill_evidence(profile: StudentProfile, skill: str, source: str) -> dict:
    """Record evidence for a skill or claim, so its issue clears on re-analysis.

    Existing entries are rebuilt as {skill, source} strings so the whole list
    passes update_profile_field's validation, and the write goes through that
    tool like any other profile change. An exact duplicate is skipped.
    """
    skill = str(skill or "").strip()
    source = str(source or "").strip()
    if not skill or not source:
        return {"status": "error", "reason": "skill and source must be non-empty"}
    entries = [
        {"skill": str(item.get("skill") or ""), "source": str(item.get("source") or "")}
        for item in profile.skill_evidence
        if isinstance(item, dict)
    ]
    new_entry = {"skill": skill, "source": source}
    if new_entry in entries:
        return {"status": "skipped", "reason": "duplicate"}
    return update_profile_field(profile, {"skill_evidence": [*entries, new_entry]})


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
        *_evidence_skills(profile),
    ]


def _evidence_skills(profile: StudentProfile) -> list[str]:
    return [
        str(item.get("skill") or "")
        for item in profile.skill_evidence
        if isinstance(item, dict)
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


def _ungrounded_edit_issues(
    sections: dict[str, str], profile: StudentProfile, job_title: str
) -> list[AnalysisIssue]:
    """Flag CV text the profile, with its evidence, does not support.

    A list item (Skills, Technologies) is itself a claim, so it must appear
    whole in the profile facts: the summary extractor skips such items when
    they are sentence-initial, lowercase or under four letters. Prose goes
    through the summary grounding check one sentence at a time.
    """
    issues: list[AnalysisIssue] = []
    for section, kind, text in _cv_units(sections):
        if kind == "item":
            if not profile_supports(text, profile):
                issues.append(_ungrounded_issue(section, text, [text]))
            continue
        terms = _unsupported_terms(text, profile, job_title)
        if not terms:
            continue
        if _ACHIEVEMENT_RE.search(text):
            issues.append(_ungrounded_achievement_issue(section, text))
        else:
            issues.append(_ungrounded_issue(section, text, terms))
    return issues


def _unsupported_terms(
    sentence: str, profile: StudentProfile, job_title: str
) -> list[str]:
    """Terms the summary check flags, minus any the profile or evidence supports."""
    return [
        term
        for term in ungrounded_terms(sentence.replace("**", ""), profile, job_title)
        if not profile_supports(term, profile)
    ]


def _ungrounded_issue(section: str, quote: str, terms: list[str]) -> AnalysisIssue:
    unique = list(dict.fromkeys(terms))
    named = _join_quoted(unique)
    single = len(unique) == 1
    it, is_, does = ("it", "is", "does") if single else ("them", "are", "do")
    return _issue(
        "ungrounded_edit",
        section,
        quote,
        "high",
        why=(
            f"{named} {is_} not in your confirmed profile, so nothing in your CV "
            f"backs {it} up."
        ),
        suggestion=(
            f"Remove {it}, or keep {it} only if you can back {it} up, for "
            "example by attaching evidence of where you did it."
        ),
        question=f"Where {does} {named} come from?",
    )


def _ungrounded_achievement_issue(section: str, quote: str) -> AnalysisIssue:
    """An achievement is one claim, so it is named as such, not word by word."""
    return _issue(
        "ungrounded_edit",
        section,
        quote,
        "high",
        why=(
            "This achievement isn't in your confirmed profile, so nothing in "
            "your CV backs it up."
        ),
        suggestion=(
            "Remove it, or keep it only if you can back it up, for example by "
            "attaching a certificate or record."
        ),
        question=(
            "Where does this achievement come from, and do you have a "
            "certificate or record for it?"
        ),
    )


def _no_evidence_issues(
    sections: dict[str, str], profile: StudentProfile, job_title: str
) -> list[AnalysisIssue]:
    """Flag listed skills and achievements that nothing in the CV shows.

    A listed skill clears when an evidence entry or any Projects/Experience
    description mentions it, compared after stemming. Unsupported skills are
    left to ungrounded_edit, so nothing is reported twice.
    """
    entries = [
        (section, item)
        for section in ITEM_SECTIONS
        for item in _parse_items(sections.get(section, ""))
    ]
    descriptions = [_stem_phrase(item.description) for _, item in entries]
    evidence = [key for key in map(_stem_phrase, _evidence_skills(profile)) if key]
    listed = [
        (section, tech, item.title)
        for section, item in entries
        for tech in item.technologies
    ]
    listed += [(SKILLS_SECTION, skill, "") for skill in _skills_items(sections)]

    issues: list[AnalysisIssue] = []
    seen: set[str] = set()
    for section, skill, entry_title in listed:
        key = _stem_phrase(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        if not profile_supports(skill, profile):
            continue
        if any(
            _has_stemmed_phrase(key, entry) or _has_stemmed_phrase(entry, key)
            for entry in evidence
        ):
            continue
        if any(_has_stemmed_phrase(text, key) for text in descriptions):
            continue
        issues.append(_listed_skill_issue(section, skill, entry_title))

    for section, kind, text in _cv_units(sections):
        if kind != "sentence" or not _ACHIEVEMENT_RE.search(text):
            continue
        if _unsupported_terms(text, profile, job_title):
            continue
        if any(_has_stemmed_phrase(_stem_phrase(text), entry) for entry in evidence):
            continue
        issues.append(
            _issue(
                "no_evidence",
                section,
                text,
                "low",
                why="This achievement is stated but not shown in your CV with any evidence.",
                suggestion=(
                    "Attach a certificate or screenshot, or say who awarded it "
                    "and when."
                ),
                question="Do you have a certificate or record for this?",
            )
        )
    return issues


def _listed_skill_issue(section: str, skill: str, entry_title: str) -> AnalysisIssue:
    if section == SKILLS_SECTION:
        return _issue(
            "no_evidence",
            section,
            skill,
            "medium",
            why=(
                f"{skill} is in your skills list but is not shown in your CV: "
                "no project or experience describes using it."
            ),
            suggestion=(
                f"Name the project, course or job where you used {skill}, or "
                "attach evidence such as a certificate."
            ),
            question=f"Where have you used {skill}?",
        )
    entry = _entry_name(section, entry_title)
    return _issue(
        "no_evidence",
        section,
        skill,
        "low",
        why=(
            f"{skill} is listed for {entry}, but no description shows how you "
            "used it, so it is not shown in your CV."
        ),
        suggestion=(
            f"If you used {skill} in {entry}, add one line on what you did with "
            "it, or attach evidence such as a screenshot."
        ),
        question=f"What did you do with {skill} in {entry}?",
    )


def _low_relevance_issues(
    sections: dict[str, str], profile: StudentProfile, job: dict | None
) -> list[AnalysisIssue]:
    """Entries with no overlap with the job, and job requirements never addressed."""
    if not job:
        return []
    description = str(job.get("description") or "")
    return [
        *_unrelated_entry_issues(sections, profile, description),
        *_unaddressed_requirement_issues(sections, job),
    ]


def _unrelated_entry_issues(
    sections: dict[str, str], profile: StudentProfile, description: str
) -> list[AnalysisIssue]:
    """Flag entries that mention none of the job skills the CV lists.

    Job skills are the CV's Skills and Technologies items that the profile
    supports and the job description mentions, the same overlap idea A1 uses
    to order Projects and Experience. With none, every entry would be
    flagged, so nothing is.
    """
    description_key = _stem_phrase(description)
    listed = [text for _, kind, text in _cv_units(sections) if kind == "item"]
    job_skills: list[str] = []
    seen: set[str] = set()
    for skill in listed:
        key = _stem_phrase(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        if profile_supports(skill, profile) and _has_stemmed_phrase(description_key, key):
            job_skills.append(skill)
    if not job_skills:
        return []

    shown = job_skills[:3]
    issues: list[AnalysisIssue] = []
    for section in ITEM_SECTIONS:
        for item in _parse_items(sections.get(section, "")):
            quote = item.bullets[0] if item.bullets else item.title
            if not quote:
                continue
            item_key = _stem_phrase(
                " ".join([item.title, item.description, *item.technologies])
            )
            if any(_has_stemmed_phrase(item_key, _stem_phrase(s)) for s in job_skills):
                continue
            entry = _entry_name(section, item.title)
            issues.append(
                _issue(
                    "low_relevance",
                    section,
                    quote,
                    "low",
                    why=(
                        f"{entry[0].upper()}{entry[1:]} doesn't mention any of the "
                        "skills this job asks for that your CV lists "
                        f"({', '.join(shown)})."
                    ),
                    suggestion=(
                        f"If this work used {_or_list(shown)}, name it and say what "
                        "you did with it. If it didn't, you don't need to change "
                        "this entry."
                    ),
                    question=f"Did {entry} involve {_or_list(shown)}?",
                )
            )
    return issues


def _unaddressed_requirement_issues(
    sections: dict[str, str], job: dict
) -> list[AnalysisIssue]:
    """Flag job requirements that no word of the CV touches.

    A requirement counts as addressed if any of its content words appears
    anywhere in the CV. That is lenient on purpose: the analyst should
    under-report gaps rather than invent them.
    """
    cv_text = " ".join(text for _, _, text in _cv_units(sections))
    cv_words = {_stem(token.lower()) for token in _WORD_RE.findall(cv_text)}
    issues: list[AnalysisIssue] = []
    for item, preferred in _job_requirements(job):
        words = _content_words(item)
        if not words or words & cv_words:
            continue
        issues.append(
            _issue(
                "low_relevance",
                JOB_AD_SECTION,
                item,
                "low" if preferred else "medium",
                why=(
                    f'The job ad lists "{item}" as preferred, and your CV doesn\'t '
                    "mention it."
                    if preferred
                    else f'The job ad asks for "{item}", and your CV doesn\'t mention it.'
                ),
                suggestion=(
                    "If you have done this in a project, course or job, add it "
                    "there. If not, leave it out rather than adding it to match."
                ),
                question=f'Have you done anything involving "{item}", even in coursework?',
            )
        )
        if len(issues) == MAX_REQUIREMENT_ISSUES:
            break
    return issues


def _job_requirements(job: dict) -> list[tuple[str, bool]]:
    """(item, preferred) pairs, required first, each item verbatim from the job.

    A requirements list from the job-ad parser is used as is. Otherwise the
    description's Requirements / Minimum qualifications blocks are required,
    Preferred / Desirable / Nice to have blocks are preferred, and other
    blocks (Responsibilities) are ignored; with no such header, the whole
    description is required. Eligibility clauses are dropped.
    """
    listed = job.get("requirements")
    if isinstance(listed, list) and listed:
        blocks = [(str(entry), False) for entry in listed]
    else:
        blocks = _requirement_blocks(str(job.get("description") or ""))

    items: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for block, preferred in blocks:
        for clause in _CLAUSE_SPLIT_RE.split(block):
            if _ELIGIBILITY_RE.search(clause):
                continue
            for part in clause.split(","):
                item = _strip_lead_ins(part.strip().rstrip("."))
                if _FILLER_ITEM_RE.match(item):
                    continue
                key = _normalize(item)
                if key and key not in seen:
                    seen.add(key)
                    items.append((item, preferred))
    return sorted(items, key=lambda pair: pair[1])


def _requirement_blocks(description: str) -> list[tuple[str, bool]]:
    headers = list(_JOB_HEADER_RE.finditer(description))
    blocks: list[tuple[str, bool]] = []
    ends = [header.start() for header in headers[1:]] + [len(description)]
    for header, end in zip(headers, ends, strict=True):
        name = header.group(0).lower()
        if "requirement" in name or "minimum" in name:
            blocks.append((description[header.end() : end], False))
        elif any(word in name for word in ("preferred", "desirable", "nice")):
            blocks.append((description[header.end() : end], True))
    return blocks or [(description, False)]


def _strip_lead_ins(item: str) -> str:
    """Drop openers such as "strong knowledge of", keeping a verbatim substring."""
    while True:
        match = _LEAD_IN_RE.match(item)
        if not match:
            return item
        item = item[match.end() :]


def _content_words(text: str) -> set[str]:
    """Stemmed words that say what a requirement is about.

    Kept: non-generic words of three or more letters, and two-letter
    acronyms (AI, ML, CI).
    """
    words: set[str] = set()
    for token in _WORD_RE.findall(text):
        lowered = token.lower()
        if lowered in _GENERIC_REQUIREMENT_WORDS:
            continue
        if len(token) < 3 and not token.isupper():
            continue
        words.add(_stem(lowered))
    return words


def _or_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} or {items[-1]}"


def _cv_units(sections: dict[str, str]) -> Iterator[tuple[str, str, str]]:
    """Every checkable unit of the CV, in order, as (section, kind, text).

    kind is "item" for each Skills or Technologies list item and "sentence"
    for each prose sentence. Text is verbatim, so it can be quoted back.
    """
    for section, body in sections.items():
        for raw_line in body.splitlines():
            line = raw_line.strip()
            tech_label = _TECH_LINE_RE.match(line)
            if tech_label or section == SKILLS_SECTION:
                if tech_label:
                    listed = line[tech_label.end() :]
                else:
                    listed = _SKILL_LABEL_RE.sub("", _BULLET_RE.sub("", line))
                for item in _split_list(listed):
                    yield section, "item", item
                continue
            text = _BULLET_RE.sub("", _HEADING_RE.sub("", line))
            for sentence in _SENTENCE_SPLIT_RE.split(text):
                if sentence.strip():
                    yield section, "sentence", sentence.strip()


def _join_quoted(terms: list[str]) -> str:
    quoted = [f'"{term}"' for term in terms]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _stem_phrase(text: str) -> str:
    """Lowercase, hyphens to spaces, edge punctuation off, -ing/-ed/-s off."""
    tokens = (token.strip(_TOKEN_EDGE_PUNCTUATION) for token in _normalize(text).split())
    return " ".join(_stem(token) for token in tokens if token)


def _stem(token: str) -> str:
    stripped = True
    while stripped:
        stripped = False
        for suffix in _STEM_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
                token = token[: -len(suffix)]
                stripped = True
                break
    return token


def _has_stemmed_phrase(text_key: str, phrase_key: str) -> bool:
    """Whole-token match between two _stem_phrase outputs."""
    if not phrase_key:
        return False
    pattern = rf"(?<!\S){re.escape(phrase_key)}(?!\S)"
    return re.search(pattern, text_key) is not None


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


def _write_feedback(
    issues: list[AnalysisIssue],
    sections: dict[str, str],
    profile: StudentProfile,
    job_title: str,
    use_gemini: bool,
) -> dict:
    """Let Gemini rewrite the issues' text; keep the template wherever it fails.

    Returns diagnostics: path (gemini, partial or template), the batch-level
    reason and detail, how many times a busy model was retried, and per
    issue_id its source (gemini or template) with the reason it fell back.
    A batch-level failure's detail is kept once, at the top level.
    """
    per_issue = {
        issue.issue_id: {"source": "template", "reason": "", "detail": ""}
        for issue in issues
    }
    batch = {"retries": 0}
    reason, detail = _gemini_rewrite(
        issues, sections, profile, job_title, use_gemini, per_issue, batch
    )
    for entry in per_issue.values():
        if entry["source"] == "template" and not entry["reason"]:
            entry["reason"] = reason
    from_gemini = sum(entry["source"] == "gemini" for entry in per_issue.values())
    if from_gemini == 0:
        path = "template"
    elif from_gemini == len(issues):
        path = "gemini"
    else:
        path = "partial"
    return {
        "path": path,
        "reason": reason,
        "detail": detail,
        "retries": batch["retries"],
        "issues": per_issue,
    }


def _gemini_rewrite(
    issues: list[AnalysisIssue],
    sections: dict[str, str],
    profile: StudentProfile,
    job_title: str,
    use_gemini: bool,
    per_issue: dict[str, dict],
    batch: dict,
) -> tuple[str, str]:
    """One batched call. Returns a batch-level (reason, detail), empty on success.

    A 503 UNAVAILABLE (the model is busy) is retried once after
    UNAVAILABLE_RETRY_SECONDS; any other error falls back at once.
    """
    if not issues:
        return "no_issues", ""
    if not use_gemini:
        return "disabled", ""
    sendable = []
    for issue in issues:
        if issue.section in _PRIVATE_SECTIONS:
            per_issue[issue.issue_id]["reason"] = "private_section"
        else:
            sendable.append(issue)
    if not sendable:
        return "private_section", ""
    client = llm_client.get_gemini_client()
    if client is None:
        return "no_client", ""

    sent = [_payload_issue(issue, sections, profile) for issue in sendable]
    request = {
        "system_instruction": _FEEDBACK_SYSTEM_INSTRUCTION,
        "user_content": json.dumps(
            {"job_title": job_title, "issues": sent}, ensure_ascii=False
        ),
        "response_schema": _FEEDBACK_RESPONSE_SCHEMA,
    }
    try:
        try:
            parsed = llm_client.call_gemini_json(client, **request)
        except llm_client.GeminiAPIError as exc:
            if not _model_unavailable(exc):
                raise
            batch["retries"] = 1
            time.sleep(UNAVAILABLE_RETRY_SECONDS)
            parsed = llm_client.call_gemini_json(client, **request)
    except llm_client.GeminiAPIError as exc:
        return "api", str(exc)
    except llm_client.GeminiJSONParseError as exc:
        return "parse", str(exc)

    returned = parsed.get("issues")
    if not isinstance(returned, list):
        return "empty", f"no issues list (got {type(returned).__name__})"
    by_id: dict[str, dict] = {}
    for item in returned:
        if isinstance(item, dict) and isinstance(item.get("issue_id"), str):
            by_id.setdefault(item["issue_id"], item)

    for issue, payload in zip(sendable, sent, strict=True):
        entry = per_issue[issue.issue_id]
        item = by_id.get(issue.issue_id)
        if item is None:
            entry["reason"] = "missing"
            continue
        fields = {name: _fill_name(item.get(name), profile) for name in FEEDBACK_FIELDS}
        problem = _feedback_problem(
            issue, payload, item.get("quote"), fields, profile, job_title
        )
        if problem is not None:
            entry["reason"], entry["detail"] = problem
            continue
        issue.why, issue.suggestion, issue.question = (
            fields[name].strip() for name in FEEDBACK_FIELDS
        )
        entry["source"] = "gemini"
    return "", ""


def _model_unavailable(exc: Exception) -> bool:
    """True for a 503 UNAVAILABLE. llm_client keeps the SDK error as the cause."""
    return getattr(exc.__cause__, "code", None) == 503


def _payload_issue(
    issue: AnalysisIssue, sections: dict[str, str], profile: StudentProfile
) -> dict:
    """What Gemini sees of one issue, with the student's name redacted."""
    context = "" if issue.section == JOB_AD_SECTION else sections.get(issue.section, "")
    return {
        "issue_id": issue.issue_id,
        "type": issue.type,
        "severity": issue.severity,
        "section": issue.section,
        "quote": _redact(issue.quote, profile),
        "context": _redact(context, profile),
        "draft": {name: _redact(getattr(issue, name), profile) for name in FEEDBACK_FIELDS},
    }


def _feedback_problem(
    issue: AnalysisIssue,
    payload: dict,
    returned_quote: object,
    fields: dict[str, object],
    profile: StudentProfile,
    job_title: str,
) -> tuple[str, str] | None:
    """Why Gemini's text for this issue cannot be used, or None if it can.

    fields are the texts the student would see, [name] already filled. The
    response must echo the quote it was given, so text cannot drift to
    another issue. Each field must be a short non-empty string with no
    banned wording and no number in digits that the issue's quote, context
    or draft does not already contain, and every claim in it must be
    supported by the profile or already appear in the issue's own quote,
    draft or section name.
    """
    echoed = str(returned_quote or "").strip().strip('"')
    if " ".join(echoed.split()) != " ".join(payload["quote"].split()):
        return "quote_mismatch", echoed[:80]
    allowed = _stem_phrase(
        " ".join([issue.quote, issue.section, *(getattr(issue, n) for n in FEEDBACK_FIELDS)])
    )
    known_numbers = _digit_numbers(
        " ".join([payload["quote"], payload["context"], *payload["draft"].values()])
    )
    for name in FEEDBACK_FIELDS:
        text = fields[name]
        if not isinstance(text, str) or not text.strip():
            return "empty_field", name
        if len(text) > MAX_FIELD_CHARS:
            return "too_long", f"{name}: {len(text)} characters"
        if _NAME_PLACEHOLDER_RE.search(text):
            return "name_placeholder", name
        banned = _BANNED_WORDING_RE.search(text)
        if banned:
            return "banned", f"{name}: {banned.group(0)}"
        invented = sorted(_digit_numbers(_PLACEHOLDER_RE.sub("", text)) - known_numbers)
        if invented:
            return "invented_number", f"{name}: {', '.join(invented)}"
        unsupported = [
            term
            for term in _unsupported_terms(text, profile, job_title)
            if not _has_stemmed_phrase(allowed, _stem_phrase(term))
        ]
        if unsupported:
            return "ungrounded", f"{name}: {', '.join(unsupported)}"
    return None


def _fill_name(text: object, profile: StudentProfile) -> object:
    """Put the student's first name where Gemini echoed [name].

    With no name on the profile the placeholder stays, and the field is
    rejected rather than shown with a placeholder in it.
    """
    first = (profile.name.split() or [""])[0]
    if not isinstance(text, str) or not first:
        return text
    return _NAME_PLACEHOLDER_RE.sub(first, text)


def _digit_numbers(text: str) -> set[str]:
    """Numbers written in digits, commas dropped: "1,000" and "1000" match."""
    return {match.group(0).replace(",", "") for match in _DIGIT_NUMBER_RE.finditer(text)}


def _redact(text: str, profile: StudentProfile) -> str:
    """Replace the student's full name and each part of it with [name]."""
    full = profile.name.strip()
    parts = {full, *full.split()}
    for part in sorted((p for p in parts if len(p) >= 2), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(part)}\b", "[name]", text)
    return text


def _sections_from_markdown(markdown: str) -> dict[str, str]:
    """Split CV markdown into editor sections, as streamlit_app does.

    Text before the first "## " heading is filed under "Summary". Kept here so
    the standalone run needs no Streamlit; a test pins it to the app's split.
    """
    sections: dict[str, str] = {}
    current = "Summary"
    buffer: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            text = "\n".join(buffer).strip()
            if text:
                sections[current] = text
            current = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    text = "\n".join(buffer).strip()
    if text:
        sections[current] = text
    return sections


def _read_jsonl(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = (json.loads(line) for line in lines if line.strip())
    return {row["document_id"]: row for row in rows}


def _format_report(
    profile: StudentProfile,
    job: dict,
    issues: list[AnalysisIssue],
    diagnostics: dict,
    show_sources: bool,
) -> str:
    counts = ", ".join(
        f"{sum(issue.severity == level for issue in issues)} {level}"
        for level in SEVERITY_ORDER
    )
    lines = [
        f"{profile.name} ({profile.document_id}) vs {job['title']} ({job['document_id']})",
        f"{len(issues)} issues: {counts}",
        f"Text: {_source_summary(diagnostics, len(issues))}",
    ]
    per_issue = diagnostics.get("issues") or {}
    for number, issue in enumerate(issues, start=1):
        lines += [
            "",
            f"{number}. [{issue.severity}] {issue.type} | {issue.section}",
            f"   quote:      {issue.quote}",
            f"   why:        {issue.why}",
            f"   suggestion: {issue.suggestion}",
            f"   question:   {issue.question}",
        ]
        if show_sources:
            source = _source_label(per_issue.get(issue.issue_id) or {})
            lines.append(f"   text:       {source}")
    return "\n".join(lines)


def _source_summary(diagnostics: dict, total: int) -> str:
    if diagnostics.get("reason") == "disabled":
        return f"template for all {total} issues (Gemini not requested; add --gemini)"
    entries = list((diagnostics.get("issues") or {}).values())
    from_gemini = sum(entry["source"] == "gemini" for entry in entries)
    summary = f"Gemini for {from_gemini} of {total} issues"
    fallbacks: dict[str, int] = {}
    for entry in entries:
        if entry["source"] == "template":
            fallbacks[entry["reason"]] = fallbacks.get(entry["reason"], 0) + 1
    if fallbacks:
        listed = ", ".join(f"{reason} {count}" for reason, count in fallbacks.items())
        summary += f"; template for {total - from_gemini} ({listed})"
    if diagnostics.get("retries"):
        summary += "; retried once after 503 UNAVAILABLE"
    if diagnostics.get("detail"):
        summary += f"; batch error: {diagnostics['detail']}"
    return summary


def _source_label(entry: dict) -> str:
    if entry.get("source") == "gemini":
        return "gemini"
    detail = f": {entry['detail']}" if entry.get("detail") else ""
    return f"template ({entry.get('reason') or 'unknown'}{detail})"


def main(argv: list[str] | None = None) -> int:
    """Analyse a demo profile's generated CV against one job and print the issues.

    The CV is built the way the app builds it, with the template summary.
    The run is deterministic, no Streamlit and no Gemini, unless --gemini asks
    for one batched Gemini call to rewrite the issues' text.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.cv_analyst",
        description="Print CV analyst issues for a demo profile and job.",
    )
    parser.add_argument("profile_id", help="e.g. student_profile_01")
    parser.add_argument("job_id", help="e.g. graduate_ai_engineer")
    parser.add_argument("--json", action="store_true", help="print the issues as JSON")
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="let Gemini rewrite the issue text (needs GEMINI_API_KEY in .env)",
    )
    args = parser.parse_args(argv)

    profiles = _read_jsonl(PROFILES_PATH)
    jobs = _read_jsonl(JOBS_PATH)
    if args.profile_id not in profiles:
        parser.error(f"unknown profile_id; choose from {', '.join(sorted(profiles))}")
    if args.job_id not in jobs:
        parser.error(f"unknown job_id; choose from {', '.join(sorted(jobs))}")

    profile = StudentProfile.model_validate(profiles[args.profile_id])
    profile.is_confirmed = True
    job = jobs[args.job_id]
    markdown = generate_cv_markdown(profile, job, job["description"])
    diagnostics: dict = {}
    issues = analyse_cv(
        profile,
        _sections_from_markdown(markdown),
        job,
        use_gemini=args.gemini,
        diagnostics=diagnostics,
    )

    if args.json:
        dumped = [issue.model_dump() for issue in issues]
        print(json.dumps(dumped, indent=2, ensure_ascii=False))
    else:
        print(_format_report(profile, job, issues, diagnostics, show_sources=args.gemini))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
