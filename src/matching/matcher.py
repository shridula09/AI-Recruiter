"""Explainable candidate-job matching and role recommendation."""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.config import WEIGHTS

ROLE_PROFILES = {
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "sql"],
    "Data Scientist": ["python", "machine learning", "statistics", "sql", "data analysis"],
    "Data Analyst": ["sql", "python", "data analysis", "statistics"],
    "NLP Engineer": ["python", "natural language processing", "machine learning"],
    "Computer Vision Engineer": ["python", "computer vision", "deep learning"],
    "Data Engineer": ["python", "sql", "spark", "data preprocessing"],
    "Backend Engineer": ["java", "sql", "docker", "rest api"],
    "ML Platform Engineer": ["python", "docker", "kubernetes", "machine learning"],
}


def _norm(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _tokens(value):
    return set(re.findall(r"[a-zA-Z0-9+#.-]+", _norm(value)))


def _similarity(a, b):
    a, b = _norm(a), _norm(b)
    if a == b:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _overlap(candidate_values, job_values, threshold=0.55):
    matched, missing = [], []
    unused = list(candidate_values)
    for required in job_values:
        best_score, best_value = max(
            ((_similarity(required, candidate), candidate) for candidate in unused),
            default=(0.0, None),
        )
        if best_score >= threshold:
            matched.append(required)
            if best_value in unused:
                unused.remove(best_value)
        else:
            missing.append(required)
    score = 100.0 * len(matched) / len(job_values) if job_values else 100.0
    return score, matched, missing


def _experience_score(candidate_years, required_years):
    if required_years <= 0:
        return 100.0
    return min(100.0, max(0.0, candidate_years / required_years * 100.0))


def _tfidf_similarity(text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([text_a, text_b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100.0)
    except ValueError:
        return 0.0


def _requirement_match(candidate, job):
    candidate_terms = (
        candidate.get("skills", [])
        + candidate.get("technologies", [])
        + candidate.get("languages", [])
    )
    required = job.get("required", [])
    preferred = job.get("preferred", [])
    required_score, matched_required, missing_required = _overlap(candidate_terms, required)
    preferred_score, matched_preferred, missing_preferred = _overlap(candidate_terms, preferred)
    return {
        "required_score": required_score,
        "preferred_score": preferred_score,
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred,
    }


def score_candidate(candidate, job):
    required = job.get("required", [])
    candidate_skills = candidate.get("skills", [])
    candidate_tech = candidate.get("technologies", [])

    skill_score, matched_skills, missing_skills = _overlap(candidate_skills, required)
    tech_score, matched_tech, missing_tech = _overlap(candidate_tech, required)
    exp_score = _experience_score(
        candidate.get("experience_years", 0),
        job.get("experience_years", 0),
    )

    project_text = " ".join(candidate.get("projects", []))
    job_text = " ".join(job.get("responsibilities", []))
    project_score = _tfidf_similarity(project_text, job_text) if project_text and job_text else 50.0

    req = _requirement_match(candidate, job)
    required_component = max(0.0, req["required_score"] - min(25.0, 5.0 * len(req["missing_required"])))

    overall = (
        WEIGHTS["skills"] * skill_score
        + WEIGHTS["technologies"] * tech_score
        + WEIGHTS["experience"] * exp_score
        + WEIGHTS["project_relevance"] * project_score
        + WEIGHTS["required_requirements"] * required_component
        + WEIGHTS["preferred_requirements"] * req["preferred_score"]
    )

    strengths = [f"Required requirement matched: {x}" for x in req["matched_required"]]
    strengths += [f"Preferred requirement matched: {x}" for x in req["matched_preferred"]]
    gaps = [f"Missing required requirement: {x}" for x in req["missing_required"]]
    gaps += [f"Missing preferred requirement: {x}" for x in req["missing_preferred"]]

    evidence = []
    for entity in candidate.get("entities", []):
        for snippet in entity.get("evidence", [])[:1]:
            evidence.append(f"{entity['name']}: {snippet}")

    return {
        "overall_score": round(max(0.0, min(100.0, overall)), 1),
        "skill_score": round(skill_score, 1),
        "technology_score": round(tech_score, 1),
        "experience_score": round(exp_score, 1),
        "project_score": round(project_score, 1),
        "required_requirement_score": round(required_component, 1),
        "preferred_requirement_score": round(req["preferred_score"], 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_technologies": matched_tech,
        "missing_technologies": missing_tech,
        "strengths": strengths[:15],
        "gaps": gaps[:15],
        "evidence": evidence[:20],
    }


def recommend_roles(candidate, top_k=5):
    terms = {
        _norm(value)
        for value in (
            candidate.get("skills", [])
            + candidate.get("technologies", [])
            + candidate.get("languages", [])
        )
    }
    result = []
    for role, requirements in ROLE_PROFILES.items():
        matched = [requirement for requirement in requirements if requirement in terms]
        score = 100.0 * len(matched) / len(requirements)
        result.append({
            "role": role,
            "score": round(score, 1),
            "matched": matched,
            "total": len(requirements),
        })
    return sorted(result, key=lambda item: item["score"], reverse=True)[:top_k]


def rank_jobs(candidate, jobs):
    scored = []
    for job in jobs:
        scored.append((score_candidate(candidate, job), job))
    return sorted(scored, key=lambda item: item[0]["overall_score"], reverse=True)
