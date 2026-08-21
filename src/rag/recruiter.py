"""Grounded recruiter assistant: retrieve evidence, build context, call Llama 3:8B."""

from src.retrieval.reranker import rerank
from src.rag.ollama import generate_stream
from src.rag.prompts import build_prompt


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [f"{key}: {val}" for key, val in value.items() if isinstance(val, (str, int, float))]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _score_summary(score):
    if not isinstance(score, dict):
        return ""
    lines = ["DETERMINISTIC MATCH RESULT"]
    labels = [
        ("overall_score", "Overall fit"),
        ("skill_score", "Skill score"),
        ("technology_score", "Technology score"),
        ("experience_score", "Experience score"),
        ("project_score", "Project score"),
        ("required_requirement_score", "Required-requirement score"),
        ("preferred_requirement_score", "Preferred-requirement score"),
    ]
    for key, label in labels:
        if key in score:
            lines.append(f"- {label}: {score[key]}")
    for title, key in [("Strengths", "strengths"), ("Gaps", "gaps")]:
        values = _as_list(score.get(key, []))
        if values:
            lines.append(f"\n{title}:")
            lines.extend(f"- {value}" for value in values[:6])
    return "\n".join(lines)


def _document_score(document):
    return float(document.get("score", 0.0)) if isinstance(document, dict) else 0.0


def build_context(query, retrieved_docs, candidate_profile, job_profile, score):
    candidate_lines = [
        f"Skills: {', '.join(candidate_profile.get('skills', []))}",
        f"Technologies: {', '.join(candidate_profile.get('technologies', []))}",
        f"Languages: {', '.join(candidate_profile.get('languages', []))}",
        f"Experience: {candidate_profile.get('experience_years', 0)} years",
        f"Projects: {', '.join(candidate_profile.get('projects', []))}",
    ]
    job_lines = [
        f"Role: {job_profile.get('title', 'Unknown')}",
        f"Required: {', '.join(job_profile.get('required', []))}",
        f"Preferred: {', '.join(job_profile.get('preferred', []))}",
        f"Experience required: {job_profile.get('experience_years', 0)} years",
        f"Responsibilities: {' '.join(job_profile.get('responsibilities', []))}",
    ]

    evidence_lines = []
    for document in retrieved_docs[:4]:
        text = document.get("text", "").strip()
        if text:
            evidence_lines.append(f"- {text[:800]}")

    return (
        f"RECRUITER QUESTION:\n{query}\n\n"
        f"CANDIDATE PROFILE:\n" + "\n".join(f"- {line}" for line in candidate_lines) + "\n\n"
        f"JOB PROFILE:\n" + "\n".join(f"- {line}" for line in job_lines) + "\n\n"
        f"{_score_summary(score)}\n\n"
        f"RETRIEVED EVIDENCE:\n" + ("\n".join(evidence_lines) if evidence_lines else "No retrieved evidence.")
    )


class RecruiterAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def _retrieve(self, query):
        documents = self.retriever.retrieve(query, k=4)
        return rerank(query, documents)[:4] if documents else []

    def build_prompt_for_question(self, query, candidate_profile, job_profile, score):
        documents = self._retrieve(query)
        context = build_context(query, documents, candidate_profile, job_profile, score)
        return build_prompt(context)

    def stream_answer(self, query, candidate_profile, job_profile, score):
        prompt = self.build_prompt_for_question(query, candidate_profile, job_profile, score)
        return generate_stream(prompt=prompt, temperature=0.1)

    def answer(self, query, candidate_profile, job_profile, score):
        return "".join(
            self.stream_answer(query, candidate_profile, job_profile, score)
        ).strip()
