"""Local job embeddings and cosine-similarity retrieval."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from app.llm_client import GeminiAPIError, GeminiJSONParseError, call_gemini_json, get_gemini_client
from app.schemas import StudentProfile

GAP_ANALYSIS_FALLBACK = "Gap analysis unavailable for this match."
_GAP_SYSTEM_INSTRUCTION = (
    "You compare a student's confirmed skills, experience, and projects against "
    "ONE specific job's actual description. Write 1-2 sentences identifying "
    "genuine gaps or strengths. ONLY reference requirements that literally appear "
    "in the provided job description text — do not invent or assume requirements "
    "not stated there. If the student's profile already covers the job well, say "
    "so honestly rather than inventing a gap. Reply with JSON only."
)
_GAP_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "gap_analysis": {"type": "STRING"},
    },
    "required": ["gap_analysis"],
}

JOBS_PATH = Path(__file__).resolve().parent.parent / "data" / "jobs.jsonl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class JobRecord(BaseModel):
    document_id: str
    title: str
    location: str | None = None
    job_type: str | None = None
    salary: str | None = None
    benefits: list[str] = Field(default_factory=list)
    description: str = ""


_jobs: list[JobRecord] | None = None
_embeddings: np.ndarray | None = None
_model = None


def _get_model():
    global _model
    if _model is None:
        started = time.perf_counter()
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(
            f"vector_store: loaded embedding model in "
            f"{time.perf_counter() - started:.3f}s",
            file=sys.stderr,
        )
    return _model


def _job_embed_text(job: JobRecord) -> str:
    return f"{job.title}\n{job.description}"


def load_jobs(force: bool = False) -> list[JobRecord]:
    """Load jobs.jsonl and cache title+description embeddings in memory."""
    global _jobs, _embeddings
    if _jobs is not None and _embeddings is not None and not force:
        return _jobs

    started = time.perf_counter()
    jobs: list[JobRecord] = []
    with JOBS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            jobs.append(JobRecord.model_validate_json(line))

    model = _get_model()
    encode_started = time.perf_counter()
    texts = [_job_embed_text(job) for job in jobs]
    embeddings = np.asarray(model.encode(texts, convert_to_numpy=True), dtype=np.float64)
    print(
        f"vector_store: encoded {len(jobs)} jobs in "
        f"{time.perf_counter() - encode_started:.3f}s "
        f"(load_jobs total {time.perf_counter() - started:.3f}s)",
        file=sys.stderr,
    )

    _jobs = jobs
    _embeddings = embeddings
    return _jobs


def build_query_text(profile: StudentProfile) -> str:
    roles = ", ".join(profile.target_roles)
    skills = ", ".join(profile.technical_skills)
    experience = " ".join(
        item.description.strip()
        for item in profile.experience
        if item.description.strip()
    )
    projects = " ".join(
        item.description.strip()
        for item in profile.projects
        if item.description.strip()
    )
    return (
        f"Target roles: {roles}. Technical skills: {skills}. "
        f"Experience: {experience}. Projects: {projects}."
    )


def search_top_jobs(profile: StudentProfile, top_k: int = 3) -> list[dict]:
    jobs = load_jobs()
    if _embeddings is None or len(jobs) == 0:
        return []

    model = _get_model()
    query = build_query_text(profile)
    query_vector = np.asarray(
        model.encode(query, convert_to_numpy=True), dtype=np.float64
    )
    scores = _cosine_scores(_embeddings, query_vector)
    ranked = sorted(
        zip(jobs, scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]

    client = get_gemini_client()
    results: list[dict] = []
    for job, score in ranked:
        match = {
            "document_id": job.document_id,
            "title": job.title,
            "similarity_score": float(score),
            "supporting_excerpt": _supporting_excerpt(job.description, profile),
        }
        if client is None:
            match["gap_analysis"] = GAP_ANALYSIS_FALLBACK
        else:
            match["gap_analysis"] = generate_gap_analysis(
                client,
                profile,
                {"title": job.title, "description": job.description},
            )
        results.append(match)
    return results


def generate_gap_analysis(client, profile: StudentProfile, job: dict) -> str:
    """Ask Gemini for a 1-2 sentence gap analysis of one job vs the profile."""
    user_content = (
        f"Job title: {job.get('title', '')}\n"
        f"Job description: {job.get('description', '')}\n"
        f"Student profile: {build_query_text(profile)}"
    )
    try:
        parsed = call_gemini_json(
            client,
            system_instruction=_GAP_SYSTEM_INSTRUCTION,
            user_content=user_content,
            response_schema=_GAP_RESPONSE_SCHEMA,
        )
    except (GeminiAPIError, GeminiJSONParseError) as exc:
        print(
            f"gap_analysis failure: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return GAP_ANALYSIS_FALLBACK

    text = parsed.get("gap_analysis")
    if not isinstance(text, str) or not text.strip():
        print(
            "gap_analysis failure: empty or missing gap_analysis key "
            f"(got {type(text).__name__}); raw_parsed={json.dumps(parsed, ensure_ascii=True)}",
            file=sys.stderr,
        )
        return GAP_ANALYSIS_FALLBACK
    return text.strip()


def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    dots = matrix @ query
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0)


def _profile_keywords(profile: StudentProfile) -> list[str]:
    tokens: list[str] = []
    for raw in [*profile.technical_skills, *profile.target_roles]:
        for token in re.findall(r"[a-zA-Z0-9]+", raw.lower()):
            if len(token) >= 3:
                tokens.append(token)
    return tokens


def _supporting_excerpt(description: str, profile: StudentProfile) -> str:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", description.strip())
        if part.strip()
    ]
    if not sentences:
        return description.strip()

    keywords = _profile_keywords(profile)
    if not keywords:
        return sentences[0]

    best_sentence = sentences[0]
    best_score = -1
    for sentence in sentences:
        lowered = sentence.lower()
        score = sum(1 for token in keywords if token in lowered)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence
