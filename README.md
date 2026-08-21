# AI Recruiter — Llama 3:8B

An explainable AI-powered recruitment assistant for the MIC AIML Department **NLP + ChatBot — AI Recruiter** challenge.

The application has three clear stages:

1. **Part 1 — Extraction:** conversational candidate input → Skills / Technologies / Languages JSON.
2. **Part 2 — Matching:** candidate profile + job description → role recommendations, match score, strengths and gaps.
3. **Part 3 — RAG Recruiter:** recruiter question → BM25 + TF-IDF retrieval → grounded context → **Llama 3:8B through Ollama**.

The architecture deliberately avoids unnecessary model layers. Part 1 and the numerical matching engine are deterministic and explainable; Llama 3:8B is used only for the required recruiter-chat generation stage.

> **Challenge compliance:** the challenge requires Llama 3:8B for Part 3 and asks the demo video to show Llama 3:8B running on the user's device. This repository is configured for the official Ollama `llama3:8b` tag. urlOllama Llama 3:8B tagshttps://ollama.com/library/llama3:8b/tags

---

## 1. Project Overview

### Part 1 — Extraction

Input is conversational/free-form text, not a structured resume. The system extracts:

- Skills
- Technologies
- Languages

The extractor uses deterministic NLP, contextual candidate discovery, phrase handling, normalization, negation filtering, technical token-shape signals and local statistical ranking. It does **not** use an LLM or pretrained NER/transformer model.

### Part 2 — Matching

The system:

- extracts candidate information from resume-style text;
- analyzes required and preferred job requirements;
- recommends suitable job roles;
- calculates an explainable candidate-job score;
- shows strengths, gaps and supporting evidence;
- compares candidates using the same deterministic matcher.

### Part 3 — RAG Recruiter

The recruiter assistant:

- retrieves relevant candidate/job evidence;
- combines BM25 and TF-IDF retrieval;
- reranks the small retrieved set lexically;
- provides the deterministic match result as grounding;
- uses Llama 3:8B locally through Ollama to generate a concise recruiter-style response.

---

## 2. Architecture

```text
                         Streamlit UI
                              |
             +----------------+----------------+
             |                                 |
       Candidate text                     Job description
             |                                 |
             v                                 v
      Text cleaning                      Job analysis
             |                                 |
             v                                 v
       Part 1 NLP --------------------> Job Profile
             |
             v
      Candidate Profile
             |
             +---------------+
                             |
                             v
                    Explainable Matcher
                             |
              +--------------+--------------+
              |              |              |
           Match score     Strengths       Gaps
                             |
                             v
                    Document construction
                             |
                             v
                    BM25 + TF-IDF retrieval
                             |
                             v
                       Lexical reranking
                             |
                             v
                    Grounded recruiter context
                             |
                             v
                     Llama 3:8B / Ollama
                             |
                             v
                    Recruiter-style answer
```

The important design choice is that **the LLM does not calculate the numerical match score**. The deterministic matcher does that first, and Llama 3:8B only explains the supplied evidence in natural language.

---

## 3. Part 1 — Conversational NLP Extraction

### Example input

```text
I worked in the AI/ML department and worked with CNN models using Python.
```

### Required output

```json
{
  "Skills": [],
  "Technologies": [],
  "Languages": []
}
```

### Extraction pipeline

```text
Text
 ↓
Cleaning + sentence segmentation
 ↓
Contextual candidate generation
 ↓
Phrase/span handling
 ↓
Context scoring
 ↓
Negation filtering
 ↓
Normalization + deduplication
 ↓
Skills / Technologies / Languages
```

The extractor also retains internal evidence and confidence information so that Parts 2 and 3 can explain where an extracted item came from.

### Why this approach?

The challenge says pretrained models are not generally allowed unless specifically mentioned. Therefore Part 1 uses deterministic NLP rather than a pretrained NER/transformer model. The approach is intentionally explainable and can be debugged one stage at a time.

### Open-vocabulary behavior

The system is not dependent on a large fixed technology master list. Technical candidates can be proposed from context and technical token shape. Small language hints and normalization rules are used as supporting signals, not as the entire extraction mechanism.

---

## 4. Part 2 — Explainable Matching

The matching engine uses simple, transparent components:

```text
Candidate + Job
      |
      +--> skill overlap
      +--> technology overlap
      +--> experience score
      +--> project relevance (TF-IDF cosine similarity)
      +--> required requirement score
      +--> preferred requirement score
      |
      v
Weighted overall score
```

The weights are stored in `config/config.py` and sum to 1.0.

The result includes:

- overall score;
- matched and missing skills/technologies;
- strengths;
- gaps;
- supporting evidence.

### Role recommendation

Role profiles are simple requirement lists. The candidate's extracted skills, technologies and languages are compared with each role profile and ranked by the percentage of requirements matched.

This is deliberately deterministic so the recruiter can see **why** a role was recommended.

---

## 5. Part 3 — RAG Recruiter

The RAG pipeline is intentionally small:

```text
Recruiter question
       |
       v
BM25 retrieval + TF-IDF retrieval
       |
       v
Rank fusion
       |
       v
Top evidence
       |
       v
Deterministic match result + profiles
       |
       v
Llama 3:8B via Ollama
       |
       v
Grounded answer
```

### Why BM25 + TF-IDF?

- **BM25** is useful for exact term matching and ranking.
- **TF-IDF cosine similarity** provides a second, transparent text-similarity signal.
- Combining the two gives more robust retrieval without introducing a vector database or embedding model that would add unnecessary complexity for this challenge.

### Why no conversation-memory module?

Conversation memory is listed as an optional stretch goal. Each recruiter question is therefore grounded independently in the candidate profile, job profile, deterministic match result and retrieved evidence. The UI may display the previous chat messages, but previous answers are not fed back to Llama 3:8B as hidden memory.

### Llama 3:8B responsibility

Llama 3:8B is used only for natural-language recruiter responses. It is instructed not to invent candidate information, recalculate scores, or make autonomous hiring decisions.

---

## 6. Streamlit Interface

### Analyze

Upload or paste candidate and job text. The application displays the required Part 1 JSON plus the richer internal profiles.

### Matching

Shows the overall fit, component scores, strengths, gaps, evidence and recommended roles.

### Recruiter Chat

Ask questions such as:

- What are the candidate's strongest areas for this role?
- What are the biggest gaps?
- Does the candidate meet the required skills?
- Why did the candidate receive this match score?
- What evidence supports the candidate's Docker experience?

Each response is grounded using retrieval before Llama 3:8B generation.

### Compare Candidates

Compares candidates against the selected job using the same deterministic scoring engine.

---

## 7. Project Structure

```text
AI-Recruiter-Llama 3:8B/
│
├── app.py
├── smoke_test.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── examples/
│   ├── sample_candidates.jsonl
│   └── sample_jobs.jsonl
│
├── docs/
│   └── PROJECT_DESIGN.md
│
└── src/
    ├── __init__.py
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── document_parser.py
    │   └── text_cleaner.py
    │
    ├── extraction/
    │   ├── __init__.py
    │   └── extractor.py
    │
    ├── job_analysis/
    │   ├── __init__.py
    │   ├── job_profiler.py
    │   └── requirement_extractor.py
    │
    ├── matching/
    │   ├── __init__.py
    │   └── matcher.py
    │
    ├── retrieval/
    │   ├── __init__.py
    │   ├── retriever.py
    │   └── reranker.py
    │
    └── rag/
        ├── __init__.py
        ├── ollama.py
        ├── prompts.py
        └── recruiter.py
```

The project was structurally consolidated so each module has a clear responsibility. The core NLP, matching and retrieval behavior is retained rather than replaced by a simpler but weaker algorithm.

---

## 8. Installation

### Prerequisites

- Python 3.10+
- Ollama
- Enough RAM/storage for the chosen local Llama 3:8B model

Ollama currently provides the official `llama3:8b` tag. The official tag listing shows  urlOllama Llama 3:8B tagshttps://ollama.com/library/llama3:8b/tags

### Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Install Llama 3:8B

```powershell
ollama pull llama3:8b
```

Verify:

```powershell
ollama list
```

The application uses Ollama locally. **No LLM API key is required.**

### Optional model override

The application defaults to `llama3:8b`. If the challenge/environment requires another Llama 3:8B Ollama tag, set:

```powershell
$env:AI_RECRUITER_MODEL="llama3:8b"
```

before running Streamlit.

---

## 9. Run

From the repository root:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL, normally:

```text
http://localhost:8501
```

Recruiter Chat requires Ollama to be running and the configured Llama 3:8B model to be installed.

---

## 10. Smoke Test

Before launching the UI, you can verify Parts 1 and 2 plus retrieval construction:

```powershell
python smoke_test.py
```

This does not call Ollama, so it can be used to check the deterministic pipeline independently.

---

## 11. Technologies

- Python
- Streamlit
- scikit-learn
- BM25 (`rank-bm25`)
- YAKE
- spaCy blank English tokenizer/sentencizer only
- PyPDF
- python-docx
- Requests
- Ollama
- Llama 3:8B

No pretrained transformer or NER model is used for Part 1 or the numerical matching engine.

---

## 12. Limitations

- Part 1 is designed for English conversational input.
- Deterministic open-vocabulary extraction cannot perfectly distinguish every unknown technical product from an arbitrary proper noun.
- Retrieval quality depends on the available candidate/job evidence.
- Llama 3:8B responses depend on the retrieved context and local model configuration.
- The match score is an explainable engineering score, not an autonomous hiring decision.
- Llama 3:8B is much larger than Llama 3:8b, so local inference requires substantially more resources.

---

## 13. Challenge Mapping

| Challenge requirement | Implementation |
|---|---|
| Conversational Part 1 input | `src/extraction/extractor.py` |
| Skills / Technologies / Languages JSON | `extract_part1_json()` |
| Resume extraction | Same Part 1 extractor |
| Job-role recommendation | `recommend_roles()` |
| Candidate-JD matching | `score_candidate()` |
| RAG recruiter | `src/retrieval/` + `src/rag/` |
| Candidate-job fit questions | Recruiter Chat |
| Reasoning/evidence | Deterministic score + retrieved evidence |
| Llama 3:8B | Ollama `llama3:8b` |
| Web application | Streamlit |

---

## 14. Responsible Use

The application is a recruiter decision-support prototype. Its output should be reviewed by a human recruiter. The LLM prompt explicitly prevents unsupported claims and autonomous hire/reject decisions.
