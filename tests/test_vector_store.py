"""Local retrieval tests. Gemini is stubbed off so gap_analysis stays the fallback."""

from __future__ import annotations

from app.vector_store import GAP_ANALYSIS_FALLBACK, search_top_jobs
from app.schemas import StudentProfile


def test_search_top_jobs_returns_top_k_ordered(
    confirmed_profile: StudentProfile,
) -> None:
    matches = search_top_jobs(confirmed_profile, top_k=3)
    assert len(matches) == 3
    scores = [match["similarity_score"] for match in matches]
    assert all(score >= 0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert all(match["document_id"] for match in matches)
    assert all(match["gap_analysis"] == GAP_ANALYSIS_FALLBACK for match in matches)


def test_search_top_jobs_respects_top_k(
    confirmed_profile: StudentProfile,
) -> None:
    matches = search_top_jobs(confirmed_profile, top_k=1)
    assert len(matches) == 1
    assert matches[0]["similarity_score"] >= 0
