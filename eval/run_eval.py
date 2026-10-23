"""Area 3 evaluation: original pipeline vs enhanced pipeline on the five demo profiles.

Three questions
  Q1  Are the analyst's flags right?
        - anchored: each flag points at a real section and quotes real CV text.
        - seeded recall: we type a skill the profile does NOT support into an
          edited section; the analyst must flag it (ground truth is known).
        - control: an edit that only repeats a supported skill must NOT be flagged.
  Q2  Are the suggestions specific?
        A suggestion is specific when it is non-empty, names something from the
        flagged text, the profile or the job, is not stock advice, and passes the
        grounding check (it adds no fact the user never gave).
  Q3  Do evidence uploads change the outcome compared with text only?
        Count "no_evidence" flags before and after a matching evidence record,
        and after an unrelated one (which must change nothing).

Run:
    python -m eval.run_eval                 # live analyser (needs Area 1 + GEMINI_API_KEY)
    python -m eval.run_eval --dry-run       # built-in fake analyser, checks the harness only
Writes eval/results/results.md and eval/results/results.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import cv_generator as cvg  # noqa: E402
from app.agent_engine import detect_thin_content, update_profile_field  # noqa: E402
from app.schemas import StudentProfile  # noqa: E402

PROFILES_PATH = ROOT / "data" / "student_profiles.jsonl"
JOBS_PATH = ROOT / "data" / "jobs.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

# Skills used to seed a fault. The first one the profile does not support is used.
SEED_SKILLS = ("Kubernetes", "Rust", "Terraform", "Kotlin", "Apache Spark")
UNRELATED_EVIDENCE = "Photo of a cat on a sofa"
STOCK_PHRASES = (
    "add more detail",
    "be more specific",
    "improve this section",
    "consider rewording",
    "make it better",
    "add more information",
)
_STOP = {"the", "and", "with", "for", "that", "this", "from", "your", "have", "into", "about"}

Analyser = Callable[[StudentProfile, dict[str, str], dict], list[Any]]


# ----------------------------------------------------------------- helpers
def _get(issue: Any, field: str) -> str:
    value = issue.get(field) if isinstance(issue, dict) else getattr(issue, field, "")
    return str(value or "")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#.]+", text.lower()) if len(w) >= 4 and w not in _STOP}


def load_profiles(limit: int | None = None) -> list[StudentProfile]:
    profiles = []
    for line in PROFILES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            profile = StudentProfile.model_validate_json(line)
            profile.is_confirmed = True
            profiles.append(profile)
    return profiles[:limit] if limit else profiles


def load_jobs() -> list[dict]:
    return [json.loads(x) for x in JOBS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


def pick_job(profile: StudentProfile, jobs: list[dict]) -> dict:
    """Job with most profile skills in its text. Keyword based, so no embedding model is needed."""
    skills = cvg._cv_skills(profile)
    best = max(jobs, key=lambda j: len(cvg._skills_mentioned_in_job(skills, j["description"])))
    return {"document_id": best["document_id"], "title": best["title"], "description": best["description"]}


def split_sections(markdown: str) -> dict[str, str]:
    import streamlit_app  # same splitter the editor uses

    return streamlit_app._split_cv_markdown(markdown)


def ungrounded(text: str, profile: StudentProfile, job_title: str) -> list[str]:
    public = getattr(cvg, "ungrounded_terms", None)  # Area 1's shared wrapper
    if public is not None:
        return list(public(text, profile, job_title))
    return list(cvg._ungrounded_summary_terms(text, profile, job_title))


def skill_of(quote: str, profile: StudentProfile) -> str:
    low = quote.lower()
    for skill in cvg._cv_skills(profile):
        if skill.lower() in low:
            return skill
    return quote.strip()


# ---------------------------------------------------------------- dry-run analyser
def fake_analyser(profile: StudentProfile, sections: dict[str, str], job: dict) -> list[dict]:
    """Tiny stand-in for Area 1, only to prove the harness runs. NOT a result."""
    issues: list[dict] = []
    for name, body in sections.items():
        if name == "Skills":
            known = cvg._normalize_for_grounding(cvg._profile_fact_blob(profile))
            items = " ".join(line.split(":", 1)[-1] for line in body.splitlines())
            for token in re.findall(r"[A-Za-z][A-Za-z+#.]{2,}", items):
                if not cvg._claim_grounded_in_blob(token, known) and token.lower() not in _STOP:
                    issues.append(dict(issue_id=f"{name}-{token}", section=name, quote=token,
                                       type="ungrounded_edit", severity="high",
                                       why=f"{token} is not in your profile.",
                                       suggestion=f"Remove {token} or tell us where you used it.",
                                       question=f"Where did you use {token}?"))
    for thin in detect_thin_content(profile):
        issues.append(dict(issue_id=f"thin-{thin['index']}", section="Projects" if thin["type"] == "project" else "Experience",
                           quote=thin["title_or_role"], type="vague", severity="medium",
                           why="Description is short.", suggestion=f"Add what you built in {thin['title_or_role']}.",
                           question="What was your part?"))
    return issues


# ------------------------------------------------------------------ the three questions
def eval_q1(profile, sections, job, issues, analyse: Analyser) -> dict:
    anchored = [
        i for i in issues
        if _get(i, "section") in sections and _get(i, "quote").lower() in sections[_get(i, "section")].lower()
    ]
    blob = cvg._normalize_for_grounding(cvg._profile_fact_blob(profile))
    fake = next((s for s in SEED_SKILLS if not cvg._claim_grounded_in_blob(s, blob)), SEED_SKILLS[-1])

    seeded = dict(sections)
    seeded["Skills"] = sections.get("Skills", "") + f", {fake}"
    seeded_issues = analyse(profile, seeded, job)
    hit = any(
        _get(i, "type") == "ungrounded_edit"
        and fake.lower() in " ".join(_get(i, f) for f in ("quote", "why", "suggestion")).lower()
        for i in seeded_issues
    )

    real = profile.technical_skills[0] if profile.technical_skills else "Python"
    control = dict(sections)
    control["Skills"] = sections.get("Skills", "") + f", {real}"
    false_positive = any(_get(i, "type") == "ungrounded_edit" for i in analyse(profile, control, job))
    return dict(flags=len(issues), anchored=len(anchored), seeded_skill=fake, seeded_caught=hit,
                control_false_positive=false_positive)


def eval_q2(profile, sections, job, issues) -> dict:
    blob = cvg._profile_fact_blob(profile) + " " + job["description"].lower()
    specific = 0
    for issue in issues:
        text = _get(issue, "suggestion")
        if not text.strip() or any(p in text.lower() for p in STOCK_PHRASES):
            continue
        anchor = _words(_get(issue, "quote")) | _words(blob)
        if not (_words(text) & anchor):
            continue
        quote = _get(issue, "quote").lower()
        # Repeating the flagged term is fine: it is the user's own text.
        if [t for t in ungrounded(text, profile, job["title"]) if t.lower() not in quote]:
            continue
        if not _get(issue, "question").strip():
            continue
        specific += 1
    return dict(suggestions=len(issues), specific=specific)


def eval_q3(profile, sections, job, issues, analyse: Analyser) -> dict:
    def count(p):
        return sum(_get(i, "type") == "no_evidence" for i in analyse(p, sections, job))

    flagged = [i for i in issues if _get(i, "type") == "no_evidence"]
    before = len(flagged)
    with_proof = profile.model_copy(deep=True)
    with_other = profile.model_copy(deep=True)
    records = []
    for issue in flagged:
        claim = _get(issue, "quote")
        skill = skill_of(claim, profile)
        for label, shows, target in (("matching", [claim], with_proof), ("unrelated", [UNRELATED_EVIDENCE], with_other)):
            ok = cvg._claim_grounded_in_blob(claim, cvg._normalize_for_grounding(" ".join(shows)))
            records.append(dict(claim=claim, skill=skill, source_name=f"sample_{label}.png", shows=shows,
                                status="supported" if ok else "not_shown"))
            if ok:  # same rule Area 2 uses: only supported claims reach skill_evidence
                update_profile_field(target, {"skill_evidence": [*target.skill_evidence,
                                                                 {"skill": skill, "source": f"sample_{label}.png"}]})
    return dict(no_evidence_before=before, after_matching=count(with_proof) if before else 0,
                after_unrelated=count(with_other) if before else 0, records=records)


# ---------------------------------------------------------------------- driver
def evaluate(profiles: list[StudentProfile], analyse: Analyser, jobs: list[dict]) -> dict:
    rows = []
    for profile in profiles:
        job = pick_job(profile, jobs)
        original_md = cvg.generate_cv_markdown(profile, job, job["description"])
        sections = split_sections(original_md)
        issues = analyse(profile, sections, job)
        rows.append(dict(
            profile=profile.name, job=job["title"],
            original=dict(flags=0, suggestions=0),  # original pipeline has no analyst
            q1=eval_q1(profile, sections, job, issues, analyse),
            q2=eval_q2(profile, sections, job, issues),
            q3=eval_q3(profile, sections, job, issues, analyse),
        ))
    return dict(date=str(date.today()), rows=rows)


def render_table(result: dict, dry_run: bool) -> str:
    rows = result["rows"]
    out = ["# Evaluation results", ""]
    if dry_run:
        out += ["> **DRY RUN: fake analyser. These numbers only prove the harness works. Do not report them.**", ""]
    out += [f"Run date: {result['date']}. Original pipeline = CV only (0 flags, 0 suggestions).", "",
            "| Profile | Job | Flags | Anchored | Seeded fault caught | Control flagged (bad) | Specific suggestions | no_evidence: text only | after matching proof | after unrelated proof |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q1, q2, q3 = r["q1"], r["q2"], r["q3"]
        job_cell = r["job"].replace("|", "/")
        out.append(f"| {r['profile']} | {job_cell} | {q1['flags']} | {q1['anchored']}/{q1['flags']} | "
                   f"{'yes' if q1['seeded_caught'] else 'NO'} ({q1['seeded_skill']}) | "
                   f"{'YES' if q1['control_false_positive'] else 'no'} | {q2['specific']}/{q2['suggestions']} | "
                   f"{q3['no_evidence_before']} | {q3['after_matching']} | {q3['after_unrelated']} |")
    n = len(rows) or 1
    flags = sum(r["q1"]["flags"] for r in rows)
    out += ["", "## Summary", "",
            f"- **Q1 flags right?** anchored {sum(r['q1']['anchored'] for r in rows)}/{flags}; "
            f"seeded fault caught {sum(r['q1']['seeded_caught'] for r in rows)}/{n}; "
            f"control wrongly flagged {sum(r['q1']['control_false_positive'] for r in rows)}/{n}.",
            f"- **Q2 specific?** {sum(r['q2']['specific'] for r in rows)}/{sum(r['q2']['suggestions'] for r in rows)} suggestions specific "
            f"(original pipeline: 0 suggestions).",
            f"- **Q3 evidence helps?** no_evidence flags {sum(r['q3']['no_evidence_before'] for r in rows)} (text only) → "
            f"{sum(r['q3']['after_matching'] for r in rows)} after matching proof, "
            f"{sum(r['q3']['after_unrelated'] for r in rows)} after unrelated proof."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="use the built-in fake analyser")
    parser.add_argument("--limit", type=int, help="only the first N profiles")
    args = parser.parse_args(argv)

    if args.dry_run:
        analyse: Analyser = fake_analyser
    else:
        try:
            from app.cv_analyst import analyse_cv
        except ImportError:
            print("app.cv_analyst is not merged yet. Use --dry-run to test the harness.", file=sys.stderr)
            return 1
        analyse = analyse_cv

    result = evaluate(load_profiles(args.limit), analyse, load_jobs())
    table = render_table(result, args.dry_run)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = "results_dry_run" if args.dry_run else "results"
    (RESULTS_DIR / f"{stem}.md").write_text(table, encoding="utf-8")
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
