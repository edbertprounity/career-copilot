"""Checks the evaluation harness itself with a scripted analyser. No live Gemini."""

from __future__ import annotations

from eval import run_eval as ev
from app.schemas import StudentProfile


def _issue(**kw):
    base = dict(issue_id="i", section="Skills", quote="Python", type="vague", severity="low",
                why="w", suggestion="", question="")
    return {**base, **kw}


def _sections(profile):
    job = ev.pick_job(profile, ev.load_jobs())
    md = ev.cvg.generate_cv_markdown(profile, job, job["description"])
    return job, ev.split_sections(md)


def test_load_profiles_gives_five_confirmed() -> None:
    profiles = ev.load_profiles()
    assert len(profiles) == 5 and all(p.is_confirmed for p in profiles)


def test_q2_counts_only_specific_grounded_suggestions() -> None:
    profile = ev.load_profiles(1)[0]
    job, sections = _sections(profile)
    issues = [
        _issue(suggestion="Add more detail", question="q?"),                     # stock advice
        _issue(suggestion="", question="q?"),                                    # empty
        _issue(suggestion="Say how you used Python in the Course Planner API.", question="q?"),  # good
        _issue(suggestion="Mention your Kubernetes clusters in the Python work.", question="q?"),  # invented fact
        _issue(suggestion="Say how you used Python in the Course Planner API.", question=""),     # no question
    ]
    assert ev.eval_q2(profile, sections, job, issues) == {"suggestions": 5, "specific": 1}


def test_q2_allows_suggestion_to_repeat_flagged_term() -> None:
    profile = ev.load_profiles(1)[0]
    job, _ = _sections(profile)
    issue = _issue(type="ungrounded_edit", quote="Kubernetes",
                   suggestion="Remove Kubernetes or say where you used it.", question="Where?")
    assert ev.eval_q2(profile, {}, job, [issue])["specific"] == 1


def test_q1_seeded_fault_and_control() -> None:
    profile = ev.load_profiles(1)[0]
    job, sections = _sections(profile)
    # Perfect analyser: flags anything in Skills that the profile does not support.
    assert ev.fake_analyser(profile, sections, job) == []
    r = ev.eval_q1(profile, sections, job, [], ev.fake_analyser)
    assert r["seeded_caught"] is True and r["control_false_positive"] is False
    # Blind analyser misses the seeded fault.
    blind = ev.eval_q1(profile, sections, job, [], lambda *_: [])
    assert blind["seeded_caught"] is False


def test_q1_anchored_requires_real_section_and_quote() -> None:
    profile = ev.load_profiles(1)[0]
    job, sections = _sections(profile)
    good = _issue(section="Skills", quote=profile.technical_skills[0])
    bad_quote = _issue(section="Skills", quote="not in the cv at all")
    bad_section = _issue(section="Nope", quote="x")
    r = ev.eval_q1(profile, sections, job, [good, bad_quote, bad_section], lambda *_: [])
    assert (r["flags"], r["anchored"]) == (3, 1)


def test_q3_matching_proof_clears_flag_unrelated_does_not() -> None:
    profile = ev.load_profiles(1)[0]
    job, sections = _sections(profile)
    skill = profile.technical_skills[0]

    def analyse(p: StudentProfile, secs, j):
        proven = any(e["source"] == "sample_matching.png" and e["skill"] == skill for e in p.skill_evidence)
        return [] if proven else [_issue(type="no_evidence", quote=skill)]

    p = profile.model_copy(deep=True)
    p.skill_evidence = []
    r = ev.eval_q3(p, sections, job, analyse(p, sections, job), analyse)
    assert (r["no_evidence_before"], r["after_matching"], r["after_unrelated"]) == (1, 0, 1)
    assert {x["status"] for x in r["records"]} == {"supported", "not_shown"}


def test_evaluate_and_table_for_dry_run() -> None:
    result = ev.evaluate(ev.load_profiles(2), ev.fake_analyser, ev.load_jobs())
    table = ev.render_table(result, dry_run=True)
    assert "DRY RUN" in table and table.count("| ") > 10
    assert "Maya Chen" in table
