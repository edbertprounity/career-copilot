"""Fact-bound, job-specific CV markdown and A4 PDF export."""

from __future__ import annotations

import re
import sys

from fpdf import FPDF

from app.llm_client import GeminiAPIError, GeminiJSONParseError, call_gemini_json
from app.schemas import ExperienceRecord, ProjectRecord, StudentProfile

PDF_ACCENT = (27, 79, 114)
PDF_SIDEBAR_FILL = (240, 244, 247)
# richness = len(technical_skills) + len(projects) + len(experience)
# two-column if richness >= LAYOUT_THRESHOLD, else single-column. Deterministic.
LAYOUT_THRESHOLD = 8
SIDEBAR_WIDTH_MM = 62
COLUMN_GUTTER_MM = 7
LAYOUT_COMMENT_RE = re.compile(r"<!--\s*cv_layout:\s*(single|two-column)\s*-->")
SIDEBAR_HEADINGS = frozenset({"Contact", "Education", "Skills"})
MAIN_HEADINGS = frozenset(
    {"Professional Summary", "Projects", "Experience", "Activities"}
)
# Contact entries render as plain "Label: value" lines, so profile link keys need
# presentable casing. Unlisted keys fall back to title case.
_CONTACT_LABELS = {
    "email": "Email",
    "github": "GitHub",
    "gitlab": "GitLab",
    "kaggle": "Kaggle",
    "linkedin": "LinkedIn",
    "phone": "Phone",
    "portfolio": "Portfolio",
    "website": "Website",
}

# Applied only to description-mined tokens. Curated technical_skills and
# project.technologies skip this filter. Lowercase generics are dropped by
# _looks_like_technical_term; this list catches Title-Case English (Document, Team).
_GENERIC_DESCRIPTION_WORDS = frozenset(
    {
        "about",
        "agent",
        "agents",
        "application",
        "applications",
        "built",
        "component",
        "components",
        "contributed",
        "created",
        "customer",
        "customers",
        "data",
        "design",
        "designed",
        "document",
        "documents",
        "environment",
        "environments",
        "evaluation",
        "experience",
        "faculty",
        "handled",
        "information",
        "interactive",
        "model",
        "models",
        "object",
        "pipeline",
        "platform",
        "platforms",
        "presented",
        "process",
        "processes",
        "project",
        "projects",
        "prototype",
        "requirements",
        "role",
        "roles",
        "scraped",
        "service",
        "services",
        "solution",
        "solutions",
        "system",
        "systems",
        "team",
        "teams",
        "testing",
        "tool",
        "tools",
        "video",
        "workflow",
        "workflows",
        "written",
    }
)

SKILL_CATEGORY_ORDER = (
    "Languages",
    "Frameworks & Tools",
    "Databases",
    "AI & Data",
    "Other Skills",
)
_SKILL_ALIASES = {
    "python": "Languages",
    "java": "Languages",
    "javascript": "Languages",
    "typescript": "Languages",
    "c++": "Languages",
    "c#": "Languages",
    "go": "Languages",
    "rust": "Languages",
    "kotlin": "Languages",
    "html": "Languages",
    "css": "Languages",
    "flask": "Frameworks & Tools",
    "django": "Frameworks & Tools",
    "react": "Frameworks & Tools",
    "docker": "Frameworks & Tools",
    "kubernetes": "Frameworks & Tools",
    "git": "Frameworks & Tools",
    "github": "Frameworks & Tools",
    "pytest": "Frameworks & Tools",
    "junit": "Frameworks & Tools",
    "jupyter": "Frameworks & Tools",
    "jupyter notebook": "Frameworks & Tools",
    "rest": "Frameworks & Tools",
    "rest apis": "Frameworks & Tools",
    "sql": "Databases",
    "sqlite": "Databases",
    "postgresql": "Databases",
    "mysql": "Databases",
    "mongodb": "Databases",
    "redis": "Databases",
    "faiss": "Databases",
    "nlp": "AI & Data",
    "rag": "AI & Data",
    "llm": "AI & Data",
    "opencv": "AI & Data",
    "yolo": "AI & Data",
    "pytorch": "AI & Data",
    "tensorflow": "AI & Data",
    "pandas": "AI & Data",
    "numpy": "AI & Data",
    "scikit-learn": "AI & Data",
    "embeddings": "AI & Data",
    "prompt design": "AI & Data",
    "tool calling": "AI & Data",
}

_SUMMARY_SYSTEM_INSTRUCTION = (
    "Write a 2-3 sentence professional summary for this student's CV, tailored "
    "to the target job. STRICT RULE: only reference degree, institution, named "
    "skills, projects, or experience that literally appear in the provided "
    "profile data. Do NOT invent, exaggerate scope or seniority, or imply "
    "experience beyond what is stated (for example do not turn a course-planner "
    "app into an enterprise platform). Target-role titles (internships or jobs "
    "the student is aiming for) must only be referenced with aspiration "
    "language: seeking, targeting, aiming for, or interested in. NEVER write "
    "'experience as', 'worked as', or any past/present employment framing for "
    "those titles — they have not been held. Reply with JSON only."
)
_SUMMARY_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"summary": {"type": "STRING"}},
    "required": ["summary"],
}
_SUMMARY_ENGLISH = frozenset(
    {
        "about",
        "across",
        "also",
        "and",
        "been",
        "being",
        "building",
        "built",
        "currently",
        "developed",
        "experience",
        "experienced",
        "for",
        "from",
        "has",
        "have",
        "her",
        "his",
        "including",
        "interested",
        "into",
        "professional",
        "profile",
        "seeking",
        "she",
        "skill",
        "skills",
        "student",
        "students",
        "that",
        "the",
        "their",
        "they",
        "this",
        "using",
        "which",
        "while",
        "who",
        "with",
    }
)
_DESCRIPTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#/-]{3,}")
_LEADING_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#/-]*")
_LEADING_PHRASE_RE = re.compile(
    r"[A-Z][A-Za-z0-9+#/-]*(?:\s+[A-Z][A-Za-z0-9+#/-]*)+"
)
_RISKY_SUMMARY_PHRASES = (
    "enterprise platform",
    "led development",
    "led a team",
    "senior engineer",
    "production platform",
)
_ROLE_EMPLOYMENT_FRAMING_RE = re.compile(
    r"(?:experience\s+in\s+the\s+role\s+of|worked\s+as|experience\s+as)"
    r"(?:\s+an?|\s+the)?\s+",
    re.IGNORECASE,
)


class CvNotConfirmedError(Exception):
    """Raised when CV generation is requested for an unconfirmed profile."""


def _log_cv_diag(message: str) -> None:
    """Write a diagnostic line to stderr. Never raise if the handle is broken."""
    try:
        print(message, file=sys.stderr)
    except OSError:
        pass


def cv_richness(profile: StudentProfile) -> int:
    """Count of technical_skills + projects + experience entries."""
    return (
        len(profile.technical_skills)
        + len(profile.projects)
        + len(profile.experience)
    )


def cv_layout(profile: StudentProfile) -> str:
    """Return 'two-column' if richness >= 8, else 'single'. Not LLM-judged."""
    if cv_richness(profile) >= LAYOUT_THRESHOLD:
        return "two-column"
    return "single"


def generate_cv_markdown(
    profile: StudentProfile,
    job_match: dict,
    job_description: str,
    professional_summary: str | None = None,
) -> str:
    """Build a job-tailored CV from confirmed profile facts only."""
    if not profile.is_confirmed:
        raise CvNotConfirmedError("profile must be confirmed before CV generation")

    skills = _cv_skills(profile)
    job_skills = _skills_mentioned_in_job(skills, job_description)
    projects = _reorder_projects(profile.projects, job_skills)
    experience = _reorder_experience(profile.experience, job_skills)
    summary = (professional_summary or "").strip() or _templated_summary(
        profile, job_description, str(job_match.get("title") or "")
    )
    layout = cv_layout(profile)

    lines: list[str] = []
    lines.append(f"# {profile.name.strip() or 'Unnamed'}")
    lines.append(f"<!-- cv_layout: {layout} -->")

    contact_lines = [
        f"{_contact_label(key)}: {_contact_value(url)}"
        for key, url in profile.links.items()
        if url
    ]
    if contact_lines:
        lines.append("")
        lines.append("## Contact")
        lines.extend(contact_lines)

    edu_parts = [
        part
        for part in (
            profile.degree.strip(),
            profile.institution.strip(),
            profile.campus.strip(),
            profile.year.strip(),
        )
        if part
    ]
    lines.append("")
    lines.append("## Education")
    if edu_parts:
        lines.append(", ".join(edu_parts))

    lines.append("")
    lines.append("## Skills")
    for category, items in _categorize_skills(skills):
        lines.append(f"**{category}:** {', '.join(items)}")

    lines.append("")
    lines.append("## Professional Summary")
    lines.append(summary)

    # Education/Skills live in the sidebar when two-column. Then Projects vs
    # Experience: whichever has more entries with overlap_count > 0 against
    # skills that also appear in THIS job is placed first. Ties keep Projects
    # before Experience. Activities always last when present. Deterministic.
    body_order = _projects_experience_order(projects, experience, job_skills)
    for section_name in body_order:
        items = projects if section_name == "Projects" else experience
        if not items:
            continue
        lines.append("")
        lines.append(f"## {section_name}")
        if section_name == "Projects":
            for project in projects:
                lines.extend(_project_lines(project))
        else:
            for item in experience:
                lines.extend(_experience_lines(item))

    activity_lines = [
        f"- {activity.strip()}"
        for activity in profile.activities
        if activity.strip()
    ]
    if activity_lines:
        lines.append("")
        lines.append("## Activities")
        lines.extend(activity_lines)

    return "\n".join(lines).strip() + "\n"


def generate_cv_summary(
    client,
    profile: StudentProfile,
    job_description: str,
    job_title: str = "",
    diagnostics: dict | None = None,
) -> str:
    """Gemini professional summary, grounded in the profile or a template fallback.

    diagnostics, when given, is filled with which path was taken and why, so a
    caller can show it. Stderr logging alone is invisible under Streamlit on
    Windows, where the diagnostic write raises OSError and is swallowed.
    """
    fallback = _templated_summary(profile, job_description, job_title)

    def record(path: str, reason: str = "", detail: str = "") -> None:
        if diagnostics is not None:
            diagnostics.update({"path": path, "reason": reason, "detail": detail})
        message = f"cv_summary path={path}"
        if reason:
            message += f" reason={reason}"
        if detail:
            message += f" {detail}"
        _log_cv_diag(message)

    user_content = (
        f"Job description:\n{job_description}\n\n"
        f"Student profile:\n{_profile_fact_blob(profile)}"
    )
    try:
        parsed = call_gemini_json(
            client,
            system_instruction=_SUMMARY_SYSTEM_INSTRUCTION,
            user_content=user_content,
            response_schema=_SUMMARY_RESPONSE_SCHEMA,
        )
    except (GeminiAPIError, GeminiJSONParseError) as exc:
        record("fallback", "api", f"{type(exc).__name__}: {exc}")
        return fallback

    text = parsed.get("summary")
    if not isinstance(text, str) or not text.strip():
        record(
            "fallback",
            "empty",
            f"missing or empty summary key (got {type(text).__name__})",
        )
        return fallback

    summary = text.strip()
    ungrounded = _ungrounded_summary_terms(summary, profile, job_title=job_title)
    if ungrounded:
        record("fallback", "ungrounded", repr(ungrounded))
        return fallback

    record("gemini")
    return summary


def generate_cv_pdf(markdown_content: str, output_path: str) -> None:
    """Write markdown CV content to an A4 PDF with explicit margins and page breaks."""
    pdf = _build_cv_pdf(markdown_content)
    pdf.output(output_path)


def _build_cv_pdf(markdown_content: str) -> FPDF:
    layout = _layout_from_markdown(markdown_content)
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    name, sections = _pdf_document_parts(markdown_content)
    full_width = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_text_color(*PDF_ACCENT)
    pdf.set_font("Helvetica", "B", 18)
    _write_wrapped(pdf, _pdf_text(name), 9, full_width)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    if layout != "two-column":
        for heading, body in sections:
            _render_pdf_heading(pdf, heading, full_width)
            for line in body:
                _render_pdf_body_line(pdf, line, full_width)
        return pdf

    sidebar = [(h, b) for h, b in sections if h in SIDEBAR_HEADINGS]
    main = [(h, b) for h, b in sections if h in MAIN_HEADINGS]
    page_left = 15.0
    sidebar_width = SIDEBAR_WIDTH_MM
    gutter = COLUMN_GUTTER_MM
    main_x = page_left + sidebar_width + gutter
    main_width = pdf.w - 15.0 - main_x
    columns_top = pdf.get_y()

    pdf.set_fill_color(*PDF_SIDEBAR_FILL)
    pdf.rect(
        page_left - 2,
        columns_top - 1,
        sidebar_width + 4,
        pdf.h - columns_top - 15 + 1,
        style="F",
    )

    pdf.set_auto_page_break(auto=False)
    _set_column_margins(pdf, page_left, sidebar_width)
    pdf.set_y(columns_top)
    for heading, body in sidebar:
        _render_pdf_heading(pdf, heading, sidebar_width)
        for line in body:
            _render_pdf_body_line(pdf, line, sidebar_width)
    sidebar_bottom = pdf.get_y()

    pdf.set_auto_page_break(auto=True, margin=15)
    _set_column_margins(pdf, main_x, main_width)
    pdf.set_y(columns_top)
    for heading, body in main:
        _render_pdf_heading(pdf, heading, main_width)
        for line in body:
            _render_pdf_body_line(pdf, line, main_width)
    main_bottom = pdf.get_y()
    main_page = pdf.page_no()

    _set_column_margins(pdf, page_left, full_width)
    if main_page == 1:
        pdf.set_y(max(sidebar_bottom, main_bottom))
    return pdf


def _set_column_margins(pdf: FPDF, left: float, width: float) -> None:
    pdf.set_left_margin(left)
    pdf.set_right_margin(pdf.w - (left + width))
    pdf.set_x(left)


def _layout_from_markdown(markdown_content: str) -> str:
    match = LAYOUT_COMMENT_RE.search(markdown_content)
    if match:
        return match.group(1)
    return "single"


def _pdf_document_parts(markdown_content: str) -> tuple[str, list[tuple[str, list[str]]]]:
    name = "Unnamed"
    sections: list[tuple[str, list[str]]] = []
    current = ""
    buffer: list[str] = []
    for raw_line in markdown_content.splitlines():
        line = LAYOUT_COMMENT_RE.sub("", raw_line).rstrip()
        if line.startswith("# "):
            name = line[2:].strip() or name
            continue
        if line.startswith("## "):
            if current:
                sections.append((current, buffer))
            current = line[3:].strip()
            buffer = []
            continue
        if not current:
            continue
        buffer.append(line)
    if current:
        sections.append((current, buffer))
    return name, sections


def _render_pdf_heading(pdf: FPDF, heading: str, usable_width: float) -> None:
    pdf.ln(2)
    pdf.set_text_color(*PDF_ACCENT)
    pdf.set_font("Helvetica", "B", 12)
    _write_wrapped(pdf, _pdf_text(heading), 7, usable_width)
    pdf.set_draw_color(*PDF_ACCENT)
    pdf.set_line_width(0.4)
    rule_y = pdf.get_y()
    pdf.line(pdf.l_margin, rule_y, pdf.l_margin + usable_width, rule_y)
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)


def _render_pdf_body_line(pdf: FPDF, raw_line: str, usable_width: float) -> None:
    line = raw_line.rstrip()
    if not line:
        pdf.ln(2)
        return
    category = re.match(r"^\*\*(.+?):\*\*\s*(.*)$", line)
    if category:
        pdf.set_font("Helvetica", "B", 10)
        _write_wrapped(pdf, _pdf_text(category.group(1) + ":"), 5, usable_width)
        rest = category.group(2).strip()
        if rest:
            pdf.set_font("Helvetica", "", 10)
            _write_wrapped(pdf, _pdf_text(rest), 5, usable_width)
        return
    if line.startswith("**"):
        pdf.set_font("Helvetica", "B", 10)
        _write_wrapped(pdf, _pdf_text(_strip_md_bold(line)), 5, usable_width)
        return
    if line.startswith("- "):
        pdf.set_font("Helvetica", "", 10)
        _write_wrapped(pdf, _pdf_text(f"* {line[2:]}"), 5, usable_width)
        return
    pdf.set_font("Helvetica", "", 10)
    _write_wrapped(pdf, _pdf_text(line), 5, usable_width)


def _write_wrapped(pdf: FPDF, text: str, height: float, usable_width: float) -> None:
    """Draw text starting at the current left margin so lines cannot walk off-page."""
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        usable_width,
        height,
        text,
        align="L",
        new_x="LMARGIN",
        new_y="NEXT",
        wrapmode=_wrap_mode(pdf, text, usable_width),
    )


def _wrap_mode(pdf: FPDF, text: str, usable_width: float) -> str:
    """Wrap at spaces, falling back to CHAR only when one token cannot fit.

    Only an unbreakable token (a long contact URL) needs character wrapping.
    Measuring the whole string instead would pick CHAR for every paragraph that
    runs past one line, which breaks ordinary words mid-character — most
    visibly in the narrow sidebar column.
    """
    for word in text.split():
        if pdf.get_string_width(word) > usable_width:
            return "CHAR"
    return "WORD"


def _contact_label(key: str) -> str:
    cleaned = key.strip().replace("_", " ").replace("-", " ")
    return _CONTACT_LABELS.get(cleaned.lower(), cleaned.title() or key.strip())


def _contact_value(url: str) -> str:
    """Drop the scheme and www prefix so a link fits one sidebar line unbroken."""
    value = url.strip()
    without_scheme = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", value)
    if without_scheme == value:
        return value
    return re.sub(r"^www\.", "", without_scheme).rstrip("/")


def _strip_md_bold(line: str) -> str:
    if line.startswith("**"):
        end = line.find("**", 2)
        if end != -1:
            return line[2:end] + line[end + 2 :]
    return line


def _cv_skills(profile: StudentProfile) -> list[str]:
    """Full genuine skill set: profile skills, project tech, description terms."""
    return [skill for skill in _skill_candidates(profile) if skill]


def _skills_mentioned_in_job(skills: list[str], job_description: str) -> list[str]:
    """Subset used only for Projects/Experience reorder and section strength."""
    haystack = job_description.lower()
    return [skill for skill in skills if skill.lower() in haystack]


def _categorize_skills(skills: list[str]) -> list[tuple[str, list[str]]]:
    buckets: dict[str, list[str]] = {name: [] for name in SKILL_CATEGORY_ORDER}
    for skill in skills:
        buckets[_skill_category(skill)].append(skill)
    return [(name, items) for name, items in buckets.items() if items]


def _skill_category(skill: str) -> str:
    key = skill.lower().strip()
    if key in _SKILL_ALIASES:
        return _SKILL_ALIASES[key]
    aliases = sorted(_SKILL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, category in aliases:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", key):
            return category
    return "Other Skills"


def _skill_candidates(profile: StudentProfile) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(text)

    known = _curated_skill_terms(profile)
    for skill in profile.technical_skills:
        add(skill)
    for project in profile.projects:
        for tech in project.technologies:
            add(tech)
        for term in _description_terms(project.description, known):
            add(term)
    for item in profile.experience:
        for term in _description_terms(item.description, known):
            add(term)
    return ordered


def _curated_skill_terms(profile: StudentProfile) -> frozenset[str]:
    """Skills and technologies the profile lists verbatim, normalized for lookup."""
    terms = {
        _normalize_for_grounding(raw)
        for raw in (
            *profile.technical_skills,
            *(tech for project in profile.projects for tech in project.technologies),
        )
        if raw.strip()
    }
    return frozenset(terms)


def _description_terms(
    text: str, known_terms: frozenset[str] = frozenset()
) -> list[str]:
    """Tech-shaped tokens mined from prose, ignoring words that open a sentence.

    "Fixed small backend issues" is capitalized because it starts a bullet, not
    because "Fixed" is a technology, so a leading word only counts when the
    profile lists it verbatim, it opens a capitalized phrase the profile lists
    (Machine Learning ...), or its shape is not plain prose (API, PyTest).
    """
    terms: list[str] = []
    for sentence in _prose_sentences(text):
        leading = _LEADING_WORD_RE.search(sentence)
        leading_start = leading.start() if leading else -1
        allow_leading = leading is not None and _leading_word_is_technical(
            sentence[leading.start() :], known_terms
        )
        for match in _DESCRIPTION_TOKEN_RE.finditer(sentence):
            token = match.group(0).rstrip(".,;:")
            if not token:
                continue
            if match.start() == leading_start and not allow_leading:
                continue
            if _looks_like_technical_term(token):
                terms.append(token)
    return terms


def _prose_sentences(text: str) -> list[str]:
    """Sentence chunks, treating line breaks and bullet markers as boundaries."""
    chunks: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]+\s+)+", "", raw_line).strip()
        if line:
            chunks.extend(_summary_sentences(line))
    return chunks


def _leading_word_is_technical(tail: str, known_terms: frozenset[str]) -> bool:
    """Judge a sentence's first word, given the sentence text from that word on."""
    match = _LEADING_WORD_RE.match(tail)
    if not match:
        return False
    token = match.group(0).rstrip(".,;:")
    if not token:
        return False
    if _normalize_for_grounding(token) in known_terms:
        return True
    letters = [ch for ch in token if ch.isalpha()]
    if letters and (
        all(ch.isupper() for ch in letters) or any(ch.isupper() for ch in letters[1:])
    ):
        return True
    if any(ch.isdigit() or ch in "+#" for ch in token):
        return True
    phrase = _LEADING_PHRASE_RE.match(tail)
    if not phrase:
        return False
    # "Implemented Java" is a verb plus a technology, not a term like "Machine
    # Learning", so a leading phrase counts only where the profile lists it.
    key = _normalize_for_grounding(phrase.group(0))
    return any(key in term for term in known_terms)


def _looks_like_technical_term(token: str) -> bool:
    """Keep description tokens that look like tech names, not generic English.

    Proper-noun / acronym shape is the main gate (Python, Flask, OpenCV, SQL).
    All-lowercase prose (team, tool, design) is dropped even if it overlaps the
    job text. Title-Case English (Document, Team) is caught by the denylist.
    Internal dots, digits, or +/# keep tokens such as Node.js or C++.
    """
    if token.lower() in _GENERIC_DESCRIPTION_WORDS:
        return False
    if any(ch.isdigit() or ch in "+#" for ch in token):
        return True
    if "." in token:
        return True
    return any(ch.isupper() for ch in token)


def _templated_summary(
    profile: StudentProfile, job_description: str = "", job_title: str = ""
) -> str:
    """Deterministic fallback summary, tailored per job without inventing anything.

    Every claim is a profile fact. The job only decides which of those facts to
    highlight: the skills this job actually mentions and the project with the
    most overlap. Without a job the wording collapses to the profile-only form.
    """
    name = profile.name.strip() or "This student"
    year = profile.year.strip() or "current"
    degree = profile.degree.strip() or "student"
    institution = profile.institution.strip() or "their institution"
    role = (
        profile.target_roles[0].strip()
        if profile.target_roles and profile.target_roles[0].strip()
        else "graduate roles"
    )
    opening = f"{name} is a {year} {degree} student at {institution}, seeking {role}."

    skills = _cv_skills(profile)
    job_skills = _skills_mentioned_in_job(skills, job_description)[:3]
    if not job_skills:
        overall = ", ".join(skills[:3]) if skills else "software development"
        return f"{opening} Technical strengths include {overall}."

    clean_title = _short_job_title(job_title)
    target = f"the {clean_title} role" if clean_title else "this role"
    detail = f"Strengths relevant to {target} include {', '.join(job_skills)}"
    evidence = _most_relevant_project(profile.projects, job_skills)
    if evidence:
        detail += f", evidenced in {evidence}"
    return f"{opening} {detail}."


def _short_job_title(job_title: str) -> str:
    """Job-board titles carry seniority tails after a pipe; keep the role part."""
    head = str(job_title).split("|")[0]
    return " ".join(head.split())


def _job_title_grounding_blob(job_title: str) -> str:
    """Normalized job title tokens that a summary may name without inventing facts."""
    raw = str(job_title or "").strip()
    if not raw:
        return ""
    return _normalize_for_grounding(f"{raw} {_short_job_title(raw)}")


def _most_relevant_project(projects: list[ProjectRecord], job_skills: list[str]) -> str:
    """Title of the highest-overlap project. Strict > keeps the earliest on ties."""
    best_title = ""
    best_score = 0
    for project in projects:
        score = _overlap_count(
            [
                project.title,
                project.context,
                project.description,
                *project.technologies,
            ],
            job_skills,
        )
        if score > best_score:
            best_score = score
            best_title = project.title.strip()
    return best_title


def _profile_fact_blob(profile: StudentProfile) -> str:
    parts = [
        profile.name,
        profile.degree,
        profile.institution,
        profile.campus,
        profile.year,
        *profile.technical_skills,
        *profile.activities,
        *profile.target_roles,
    ]
    for project in profile.projects:
        parts.extend(
            [project.title, project.context, project.description, *project.technologies]
        )
    for item in profile.experience:
        parts.extend([item.role, item.organisation, item.type, item.description])
    for key, url in profile.links.items():
        parts.extend([key, url])
    return " ".join(part for part in parts if part).lower()


def _summary_sentences(summary: str) -> list[str]:
    """Split on .!? + whitespace + capital so claims cannot cross sentences."""
    text = summary.strip()
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        if part.strip()
    ]


def _extract_summary_claims(
    summary: str,
    skills: list[str],
    profile_blob: str = "",
    job_title_blob: str = "",
) -> list[str]:
    """Proper-noun / tech claims, extracted per sentence.

    The first word of a sentence is not a claim unless it is a known skill, or
    it starts a capitalized phrase that already appears in the profile
    (Team Course Planner) or is itself tech-shaped (REST APIs). Job-title
    tokens are not claims — naming the target role is not inventing a skill.
    """
    claims: list[str] = []
    seen: set[str] = set()
    skill_keys = {_normalize_for_grounding(skill) for skill in skills}

    def add(raw: str) -> None:
        text = raw.strip().rstrip(".,;:")
        if not text:
            return
        key = _normalize_for_grounding(text)
        if key in seen or key in _SUMMARY_ENGLISH:
            return
        if job_title_blob and _claim_grounded_in_blob(text, job_title_blob):
            return
        seen.add(key)
        claims.append(text)

    def is_known_skill(token: str) -> bool:
        return _normalize_for_grounding(token) in skill_keys

    def phrase_in_profile(phrase: str) -> bool:
        return bool(profile_blob) and _normalize_for_grounding(phrase) in profile_blob

    for skill in sorted(skills, key=len, reverse=True):
        if _normalize_for_grounding(skill) in _normalize_for_grounding(summary):
            add(skill)
    # Do not allow '.' inside a multi-word phrase — that is how "PyTest. Her" merged.
    phrase_token = r"[A-Z][A-Za-z0-9+#/-]*"
    phrase_re = re.compile(rf"\b{phrase_token}(?:\s+{phrase_token})+\b")
    token_re = re.compile(r"[A-Za-z][A-Za-z0-9+.#/-]{3,}")
    first_word_re = re.compile(r"[A-Za-z][A-Za-z0-9+.#/-]*")
    for sentence in _summary_sentences(summary):
        stripped = sentence.lstrip()
        first_match = first_word_re.match(stripped)
        first_key = (
            _normalize_for_grounding(first_match.group(0).rstrip(".,;:"))
            if first_match
            else ""
        )
        kept_initial_phrase = False
        for phrase in phrase_re.findall(sentence):
            phrase_first = phrase.split()[0]
            phrase_is_initial = (
                first_key
                and _normalize_for_grounding(phrase_first) == first_key
            )
            if phrase_is_initial and not phrase_in_profile(phrase):
                letters = [ch for ch in phrase_first if ch.isalpha()]
                internal_cap = any(ch.isupper() for ch in letters[1:]) if letters else False
                all_caps = bool(letters) and all(ch.isupper() for ch in letters)
                if not is_known_skill(phrase_first) and not internal_cap and not all_caps:
                    # "With Docker" is English + a later term, not "REST APIs".
                    continue
            add(phrase)
            if phrase_is_initial:
                kept_initial_phrase = True
        for raw in token_re.findall(sentence):
            token = raw.rstrip(".,;:")
            if first_key and _normalize_for_grounding(token) == first_key:
                if not kept_initial_phrase and not is_known_skill(token):
                    continue
            letters = [ch for ch in token if ch.isalpha()]
            if not letters:
                continue
            internal_cap = any(ch.isupper() for ch in letters[1:])
            all_caps = all(ch.isupper() for ch in letters)
            title_case = letters[0].isupper() and all(ch.islower() for ch in letters[1:])
            if internal_cap or all_caps or (title_case and len(token) >= 4):
                add(token)
    return claims


def _ungrounded_summary_terms(
    summary: str, profile: StudentProfile, job_title: str = ""
) -> list[str]:
    """Return claim terms in the summary that do not appear in the profile blob.

    Words from job_title are allowed: a summary may name the role it is
    tailored for. Fabricated skills and experience still fail.
    """
    blob = _normalize_for_grounding(_profile_fact_blob(profile))
    job_blob = _job_title_grounding_blob(job_title)
    skills = _cv_skills(profile)
    claims = _extract_summary_claims(summary, skills, blob, job_blob)

    ungrounded: list[str] = []
    for claim in claims:
        if job_blob and _claim_grounded_in_blob(claim, job_blob):
            continue
        if not _claim_grounded_in_blob(claim, blob):
            ungrounded.append(claim)
    summary_norm = _normalize_for_grounding(summary)
    for phrase in _RISKY_SUMMARY_PHRASES:
        if phrase in summary_norm and phrase not in blob:
            ungrounded.append(phrase)
    ungrounded.extend(_target_role_framing_violations(summary, profile))
    return ungrounded


def _held_experience_blob(profile: StudentProfile) -> str:
    """Projects and work history only — not target_roles or other profile fields."""
    parts: list[str] = []
    for project in profile.projects:
        parts.extend(
            [project.title, project.context, project.description, *project.technologies]
        )
    for item in profile.experience:
        parts.extend([item.role, item.organisation, item.type, item.description])
    return _normalize_for_grounding(" ".join(part for part in parts if part))


def _aspiration_only_roles(profile: StudentProfile) -> list[str]:
    """target_roles that do not also appear in projects or experience."""
    held = _held_experience_blob(profile)
    roles: list[str] = []
    seen: set[str] = set()
    for role in profile.target_roles:
        text = str(role).strip()
        if not text:
            continue
        key = _normalize_for_grounding(text)
        if key in seen:
            continue
        if held and _claim_grounded_in_blob(text, held):
            continue
        seen.add(key)
        roles.append(text)
    return roles


def _consume_role_prefix(text: str, role: str) -> str | None:
    """If text starts with role (normalized), return the unmatched tail."""
    remaining_words = text.split()
    role_words = _normalize_for_grounding(role).split()
    if not role_words or len(remaining_words) < len(role_words):
        return None
    head = _normalize_for_grounding(" ".join(remaining_words[: len(role_words)]))
    if head != _normalize_for_grounding(role):
        return None
    return " ".join(remaining_words[len(role_words) :])


def _target_role_framing_violations(
    summary: str, profile: StudentProfile
) -> list[str]:
    """Flag employment framing immediately before aspiration-only target roles."""
    roles = _aspiration_only_roles(profile)
    if not roles:
        return []
    ranked = sorted(roles, key=lambda item: len(_normalize_for_grounding(item)), reverse=True)
    flagged: list[str] = []
    seen: set[str] = set()
    for match in _ROLE_EMPLOYMENT_FRAMING_RE.finditer(summary):
        rest = summary[match.end() :]
        sentence_end = re.search(r"[.!?]", rest)
        clause = rest if sentence_end is None else rest[: sentence_end.start()]
        remaining = " ".join(clause.split())
        while remaining:
            remaining = re.sub(r"^(?:,|and|&)\s+", "", remaining, flags=re.IGNORECASE)
            remaining = re.sub(r"^(?:an?|the)\s+", "", remaining, flags=re.IGNORECASE)
            remaining = remaining.strip()
            if not remaining:
                break
            hit = None
            leftover = None
            for role in ranked:
                leftover = _consume_role_prefix(remaining, role)
                if leftover is not None:
                    hit = role
                    break
            if hit is None:
                break
            key = _normalize_for_grounding(hit)
            if key not in seen:
                seen.add(key)
                flagged.append(f"experience as {hit}")
            remaining = leftover
    return flagged


def _normalize_for_grounding(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def _claim_grounded_in_blob(claim: str, blob: str) -> bool:
    norm = _normalize_for_grounding(claim)
    if norm in blob:
        return True
    tokens = [
        token
        for token in norm.split()
        if token not in _SUMMARY_ENGLISH and len(token) >= 3
    ]
    if not tokens:
        return True
    return all(token in blob for token in tokens)


def _projects_experience_order(
    projects: list[ProjectRecord],
    experience: list[ExperienceRecord],
    skills: list[str],
) -> tuple[str, str]:
    project_strength = sum(
        1
        for project in projects
        if _overlap_count(
            [project.title, project.context, project.description, *project.technologies],
            skills,
        )
        > 0
    )
    experience_strength = sum(
        1
        for item in experience
        if _overlap_count(
            [item.role, item.organisation, item.type, item.description],
            skills,
        )
        > 0
    )
    if experience_strength > project_strength:
        return ("Experience", "Projects")
    return ("Projects", "Experience")


def _reorder_projects(
    projects: list[ProjectRecord], matched_skills: list[str]
) -> list[ProjectRecord]:
    ranked = [
        (
            index,
            -_overlap_count(
                [project.title, project.context, project.description, *project.technologies],
                matched_skills,
            ),
            project,
        )
        for index, project in enumerate(projects)
    ]
    ranked.sort(key=lambda item: (item[1], item[0]))
    return [item[2] for item in ranked]


def _reorder_experience(
    experience: list[ExperienceRecord], matched_skills: list[str]
) -> list[ExperienceRecord]:
    ranked = [
        (
            index,
            -_overlap_count(
                [item.role, item.organisation, item.type, item.description],
                matched_skills,
            ),
            item,
        )
        for index, item in enumerate(experience)
    ]
    ranked.sort(key=lambda item: (item[1], item[0]))
    return [item[2] for item in ranked]


def _overlap_count(parts: list[str], matched_skills: list[str]) -> int:
    blob = " ".join(parts).lower()
    return sum(1 for skill in matched_skills if skill.lower() in blob)


def _project_lines(project: ProjectRecord) -> list[str]:
    title = project.title.strip() or "Untitled project"
    line = f"**{title}**"
    if project.context.strip():
        line += f" ({project.context.strip()})"
    lines = [line]
    if project.description.strip():
        lines.append(f"- {project.description.strip()}")
    if project.technologies:
        lines.append(f"- Technologies: {', '.join(project.technologies)}")
    return lines


def _experience_lines(item: ExperienceRecord) -> list[str]:
    header = ", ".join(
        part
        for part in (item.role.strip(), item.organisation.strip(), item.type.strip())
        if part
    )
    lines = [f"**{header or 'Experience'}**"]
    if item.description.strip():
        lines.append(f"- {item.description.strip()}")
    return lines


def _pdf_text(value: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    text = value
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")
