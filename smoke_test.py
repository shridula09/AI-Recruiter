"""Small local smoke test for Parts 1 and 2 plus retrieval construction."""

from src.extraction.extractor import build_candidate_profile, extract_part1_json
from src.job_analysis.job_profiler import build_job_profile
from src.matching.matcher import recommend_roles, score_candidate
from src.retrieval.retriever import HybridRetriever, build_documents

CANDIDATE = "I have 2 years of experience in Python, SQL and machine learning. I built a CNN project using TensorFlow and Docker."
JOB = "Machine Learning Engineer. Required skills: Python, machine learning, SQL. Preferred skills: PyTorch, Docker. 2 years of experience preferred."

part1 = extract_part1_json(CANDIDATE)
assert set(part1) == {"Skills", "Technologies", "Languages"}

candidate = build_candidate_profile(CANDIDATE, "SMOKE")
job = build_job_profile(JOB, "JOB")
score = score_candidate(candidate, job)
roles = recommend_roles(candidate)
documents = build_documents(candidate, job)
retriever = HybridRetriever(documents)
retrieved = retriever.retrieve("Does the candidate know Python?", k=3)

assert isinstance(score["overall_score"], float)
assert roles
assert documents
assert retrieved
print("Smoke test passed.")
print("Part 1:", part1)
print("Overall score:", score["overall_score"])
print("Top role:", roles[0]["role"], roles[0]["score"])
print("Retrieved documents:", len(retrieved))
