import os

# The challenge explicitly requires Llama 3:8B for Part 3.
# The default model is the required local Ollama Llama 3:8B tag.
LLM_MODEL = os.getenv("AI_RECRUITER_MODEL", "llama3:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Explainable score weights; they sum to 1.0.
WEIGHTS = {
    "skills": 0.25,
    "technologies": 0.15,
    "experience": 0.15,
    "project_relevance": 0.15,
    "required_requirements": 0.25,
    "preferred_requirements": 0.05,
}

MAX_RETRIEVED_DOCUMENTS = 6
