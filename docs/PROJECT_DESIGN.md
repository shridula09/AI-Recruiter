# AI Recruiter — Project Design

This document is intentionally short and maps the implementation to the three challenge parts so the architecture can be explained clearly in a technical discussion.

## 1. Architecture

```text
Candidate text ──> Part 1 Extraction ──> Candidate Profile ──┐
                                                             │
Job description ─> Job Analysis ─────────> Job Profile ──────┤
                                                             v
                                                   Explainable Matcher
                                                             │
                                      +----------------------+----------------+
                                      |                      |                |
                                  Match score             Strengths          Gaps
                                      |
                                      v
                                RAG document set
                                      |
                                      v
                              BM25 + TF-IDF retrieval
                                      |
                                      v
                                 Reranking
                                      |
                                      v
                              Grounded context
                                      |
                                      v
                            Llama 3:8B / Ollama
                                      |
                                      v
                              Recruiter answer
```

## 2. Part 1 — Extraction

`src/extraction/extractor.py` contains the complete deterministic extraction pipeline.

The main stages are:

1. clean and segment the text;
2. generate contextual candidate phrases/spans;
3. use technical orthographic signals and local statistical signals to rank candidates;
4. classify candidates using contextual cues for Skill / Technology / Language / Project;
5. apply negation and clause-boundary checks;
6. resolve overlapping spans;
7. normalize and deduplicate results;
8. return the required JSON plus internal evidence.

The implementation does not load a pretrained NER or transformer model.

### Why keep the statistical signals?

YAKE-style/local statistical scoring and TF-IDF provide useful ranking evidence, but they do not directly decide the final category. The final category is determined by the project's contextual scoring rules. This keeps the result explainable.

## 3. Candidate and Job Profiles

The candidate profile is a reusable representation containing:

- skills;
- technologies;
- languages;
- projects;
- experience years;
- evidence;
- source text.

The job profile contains:

- title;
- required requirements;
- preferred requirements;
- experience requirement;
- responsibilities;
- requirement sentences;
- evidence;
- source text.

This separation makes Part 2 and Part 3 independent of the UI.

## 4. Part 2 — Matching

`src/matching/matcher.py` contains the complete scoring logic.

The final score combines:

- skill overlap;
- technology overlap;
- experience fit;
- project relevance using TF-IDF cosine similarity;
- required requirement coverage;
- preferred requirement coverage.

The weights are visible in `config/config.py`.

The matcher also produces strengths, gaps and evidence. This is important because the recruiter should be able to see why the score was produced rather than receiving only one opaque number.

Role recommendation uses a small set of role profiles and ranks them by the proportion of role requirements matched.

## 5. Part 3 — Retrieval

`src/retrieval/retriever.py` contains:

- document construction;
- BM25 retrieval;
- TF-IDF retrieval;
- reciprocal-rank fusion.

`src/retrieval/reranker.py` applies a small lexical reranking step to the already small candidate set.

### Why no vector database?

The challenge does not require one, and the application has a small local document set. BM25 + TF-IDF gives transparent retrieval without adding another infrastructure layer.

## 6. Part 3 — Llama 3:8B

`src/rag/ollama.py` handles only communication with the local Ollama server.

`src/rag/prompts.py` contains the grounding instructions.

`src/rag/recruiter.py` performs:

```text
question
 -> retrieve
 -> rerank
 -> construct context
 -> call Llama 3:8B
 -> stream answer
```

The deterministic score is calculated before the LLM call. Llama 3:8B is therefore not responsible for numerical candidate scoring.

The project defaults to the official Ollama `llama3:8b` tag because the challenge explicitly requires Llama 3:8B. urlOllama Llama 3:8B tagshttps://ollama.com/library/llama3:8b/tags

## 7. Why the Structure Is Deliberately Small

The project originally separated many small helpers. They have been consolidated into meaningful modules without removing the underlying extraction, matching or retrieval techniques.

The final mental model is only:

```text
Preprocessing
Extraction
Job Analysis
Matching
Retrieval
RAG
```

That is easier to debug, present and defend in an interview while preserving the core behavior.

## 8. No Conversation Memory

Conversation memory is not implemented as an LLM context feature. The UI may display previous messages, but each question is independently grounded using the current candidate, job, score and retrieved evidence.

This keeps the required system deterministic and avoids an unnecessary state-management layer.
