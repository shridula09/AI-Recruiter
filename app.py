import streamlit as st

from src.preprocessing.document_parser import parse_document
from src.extraction.extractor import build_candidate_profile
from src.job_analysis.job_profiler import build_job_profile
from src.matching.matcher import score_candidate
from src.matching.matcher import recommend_roles
from src.retrieval.retriever import build_documents
from src.retrieval.retriever import HybridRetriever
from src.rag.recruiter import RecruiterAgent
from src.rag.ollama import check_ollama


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Recruiter",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Recruiter")

st.caption(
    "Open-vocabulary NLP extraction + explainable matching "
    "+ hybrid retrieval + local Llama 3:8B"
)


# ============================================================
# SAMPLE DATA
# ============================================================

DEFAULT_CANDIDATE = """Computer Science graduate with 2 years of experience in Python, SQL and machine learning.

Built a CNN image classification project using TensorFlow and worked on data preprocessing,
model deployment with Docker."""


DEFAULT_JOB = """Machine Learning Engineer.

Required skills: Python, machine learning, SQL.

Preferred skills: PyTorch, Docker.

The role requires building and evaluating machine learning models,
preparing data and deploying production models.

2 years of relevant experience is preferred."""


# ============================================================
# SESSION STATE
# ============================================================

if "candidate_profile" not in st.session_state:
    st.session_state.candidate_profile = None

if "job_profile" not in st.session_state:
    st.session_state.job_profile = None

if "score" not in st.session_state:
    st.session_state.score = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chat" not in st.session_state:
    st.session_state.chat = []


# ============================================================
# TABS
# ============================================================

tab_analyze, tab_matching, tab_chat, tab_compare = st.tabs(
    [
        "Analyze",
        "Matching",
        "Recruiter Chat",
        "Compare Candidates",
    ]
)


# ################################################################
# ANALYZE
# ################################################################

with tab_analyze:

    st.header(
        "Candidate & Job Analysis"
    )

    left, right = st.columns(2)

    # ============================================================
    # CANDIDATE
    # ============================================================

    with left:

        st.subheader(
            "Candidate"
        )

        candidate_upload = st.file_uploader(
            "Upload candidate TXT / PDF / DOCX",
            type=[
                "txt",
                "pdf",
                "docx",
            ],
            key="candidate_upload",
        )

        candidate_text = st.text_area(
            "Or paste candidate information",
            value=DEFAULT_CANDIDATE,
            height=250,
            key="candidate_text",
        )

        if candidate_upload:

            try:

                candidate_text = parse_document(
                    candidate_upload
                )

                st.success(
                    f"Loaded: {candidate_upload.name}"
                )

            except Exception as exc:

                st.error(
                    f"Could not read candidate document: {exc}"
                )

    # ============================================================
    # JOB
    # ============================================================

    with right:

        st.subheader(
            "Job Description"
        )

        job_upload = st.file_uploader(
            "Upload job TXT / PDF / DOCX",
            type=[
                "txt",
                "pdf",
                "docx",
            ],
            key="job_upload",
        )

        job_text = st.text_area(
            "Or paste job description",
            value=DEFAULT_JOB,
            height=250,
            key="job_text",
        )

        if job_upload:

            try:

                job_text = parse_document(
                    job_upload
                )

                st.success(
                    f"Loaded: {job_upload.name}"
                )

            except Exception as exc:

                st.error(
                    f"Could not read job document: {exc}"
                )

    st.divider()

    # ============================================================
    # ANALYZE
    # ============================================================

    if st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Analyzing candidate and job..."
            ):

                candidate = build_candidate_profile(
                    candidate_text,
                    "LIVE",
                    None,
                )

                job = build_job_profile(
                    job_text,
                    "LIVE",
                    None,
                )

                score = score_candidate(
                    candidate,
                    job,
                )

                documents = build_documents(
                    candidate,
                    job,
                )

            st.session_state.candidate_profile = candidate
            st.session_state.job_profile = job
            st.session_state.score = score
            st.session_state.documents = documents

            # New candidate/job = new conversation.
            st.session_state.chat = []

            st.success(
                "Candidate and job analyzed successfully."
            )

        except Exception as exc:

            st.error(
                "Analysis failed."
            )

            st.exception(exc)

    # ============================================================
    # PROFILE OUTPUT
    # ============================================================

    if st.session_state.candidate_profile:

        st.divider()

        st.subheader(
            "Part 1 — Required JSON"
        )

        candidate = (
            st.session_state.candidate_profile
        )

        st.json(
            {
                "Skills": candidate.get(
                    "skills",
                    [],
                ),
                "Technologies": candidate.get(
                    "technologies",
                    [],
                ),
                "Languages": candidate.get(
                    "languages",
                    [],
                ),
            }
        )

        st.subheader(
            "Candidate Profile"
        )

        st.json(
            st.session_state.candidate_profile
        )

        st.subheader(
            "Job Profile"
        )

        st.json(
            st.session_state.job_profile
        )


# ################################################################
# MATCHING
# ################################################################

with tab_matching:

    st.header(
        "Candidate–Job Matching"
    )

    if not st.session_state.candidate_profile:

        st.info(
            "Analyze a candidate and job first."
        )

    else:

        score = st.session_state.score

        st.metric(
            "Overall Fit",
            f"{score['overall_score']}/100",
        )

        st.divider()

        score_keys = [
            "skill_score",
            "technology_score",
            "experience_score",
            "project_score",
            "required_requirement_score",
            "preferred_requirement_score",
        ]

        columns = st.columns(
            len(score_keys)
        )

        for column, key in zip(
            columns,
            score_keys,
        ):

            label = (
                key
                .replace(
                    "_score",
                    "",
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

            with column:

                st.metric(
                    label,
                    f"{score[key]:.1f}",
                )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Strengths"
            )

            strengths = score.get(
                "strengths",
                [],
            )

            if strengths:

                for item in strengths:

                    st.success(
                        item
                    )

            else:

                st.write(
                    "No strengths identified."
                )

            st.subheader(
                "Gaps"
            )

            gaps = score.get(
                "gaps",
                [],
            )

            if gaps:

                for item in gaps:

                    st.warning(
                        item
                    )

            else:

                st.write(
                    "No major gaps identified."
                )

        with right:

            st.subheader(
                "Evidence"
            )

            evidence = score.get(
                "evidence",
                [],
            )

            if evidence:

                for item in evidence:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(
                    "No evidence available."
                )

        st.divider()

        st.subheader(
            "Recommended Roles"
        )

        try:

            roles = recommend_roles(
                st.session_state.candidate_profile
            )

            if roles:

                for role in roles:

                    st.write(
                        f"**{role['role']}** — "
                        f"{role['score']}/100"
                    )

            else:

                st.info(
                    "No role recommendations available."
                )

        except Exception as exc:

            st.warning(
                f"Could not generate role recommendations: {exc}"
            )


# ################################################################
# RECRUITER CHAT
# ################################################################

with tab_chat:

    st.header(
        "💬 Recruiter Chat"
    )

    st.caption(
        "Ask grounded questions about the candidate, "
        "job, match result, strengths, gaps and evidence."
    )

    # ============================================================
    # CONTEXT STATUS
    # ============================================================

    st.subheader(
        "Chat Context"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        if st.session_state.candidate_profile:

            st.success(
                "✓ Candidate loaded"
            )

        else:

            st.error(
                "✗ Candidate not analyzed"
            )

    with c2:

        if st.session_state.job_profile:

            st.success(
                "✓ Job loaded"
            )

        else:

            st.error(
                "✗ Job not analyzed"
            )

    with c3:

        if st.session_state.score:

            st.success(
                "✓ Match calculated"
            )

        else:

            st.error(
                "✗ Match unavailable"
            )

    # ============================================================
    # OLLAMA
    # ============================================================

    st.subheader(
        "Local Model Status"
    )

    health = check_ollama()

    if health["ok"] and health["model_installed"]:

        st.success(
            f"✓ Ollama connected — "
            f"{health['model']} is available"
        )

    elif health["ok"]:

        st.warning(
            f"Ollama is running, but "
            f"{health['model']} is not installed."
        )

        st.code(
            f"ollama pull {health['model']}",
            language="powershell",
        )

    else:

        st.error(
            "Ollama is not reachable."
        )

        st.write(
            health["message"]
        )

    st.divider()

    # ============================================================
    # REQUIRE ANALYSIS
    # ============================================================

    if not st.session_state.candidate_profile:

        st.info(
            "Analyze a candidate and job first."
        )

    else:

        # ========================================================
        # EXAMPLE QUESTIONS
        # ========================================================

        st.subheader(
            "Try one of these"
        )

        examples = [
            "What are the candidate's strongest areas for this role?",
            "What are the biggest gaps?",
            "Does the candidate meet the required skills?",
            "Which preferred requirements does the candidate satisfy?",
            "Why did the candidate receive this match score?",
            "What evidence supports the candidate's Docker experience?",
            "What should a recruiter verify during an interview?",
        ]

        example_columns = st.columns(2)

        # Store selected example temporarily.
        if (
            "selected_example"
            not in st.session_state
        ):

            st.session_state.selected_example = None

        for index, example in enumerate(
            examples
        ):

            with example_columns[
                index % 2
            ]:

                if st.button(
                    example,
                    key=f"example_{index}",
                    use_container_width=True,
                ):

                    st.session_state.selected_example = (
                        example
                    )

        # ========================================================
        # DISPLAY HISTORY
        # ========================================================

        for message in st.session_state.chat:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        # ========================================================
        # INPUT
        # ========================================================

        typed_query = st.chat_input(
            "Ask about the candidate, job, score, gaps or evidence..."
        )

        selected_example = (
            st.session_state.selected_example
        )

        # Consume the example.
        st.session_state.selected_example = None

        query = (
            typed_query
            if typed_query
            else selected_example
        )

        # ========================================================
        # GENERATE
        # ========================================================

        if query and query.strip():

            query = query.strip()

            # ----------------------------------------------------
            # USER MESSAGE
            # ----------------------------------------------------

            st.session_state.chat.append(
                {
                    "role": "user",
                    "content": query,
                }
            )

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    query
                )

            # ----------------------------------------------------
            # ASSISTANT
            # ----------------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                answer_placeholder = st.empty()

                status_placeholder = st.empty()

                try:

                    # ------------------------------------------------
                    # DOCUMENTS
                    # ------------------------------------------------

                    documents = (
                        st.session_state.documents
                    )

                    if not documents:

                        status_placeholder.info(
                            "Preparing recruiter context..."
                        )

                        documents = build_documents(
                            st.session_state.candidate_profile,
                            st.session_state.job_profile,
                        )

                        st.session_state.documents = (
                            documents
                        )

                    # ------------------------------------------------
                    # RETRIEVER
                    # ------------------------------------------------

                    status_placeholder.info(
                        "🔎 Retrieving relevant evidence..."
                    )

                    retriever = HybridRetriever(
                        documents
                    )

                    # ------------------------------------------------
                    # AGENT
                    # ------------------------------------------------

                    agent = RecruiterAgent(
                        retriever
                    )

                    # ------------------------------------------------
                    # LLM
                    # ------------------------------------------------

                    status_placeholder.info(
                        "🧠 Llama 3:8B is generating..."
                    )

                    chunks = []

                    stream = agent.stream_answer(
                        query=query,
                        candidate_profile=(
                            st.session_state.candidate_profile
                        ),
                        job_profile=(
                            st.session_state.job_profile
                        ),
                        score=(
                            st.session_state.score
                        ),
                    )

                    for chunk in stream:

                        if chunk:

                            chunks.append(
                                chunk
                            )

                            answer_placeholder.markdown(
                                "".join(
                                    chunks
                                )
                            )

                    answer = "".join(
                        chunks
                    ).strip()

                    if not answer:

                        raise RuntimeError(
                            "Llama 3:8B returned an empty response."
                        )

                    status_placeholder.empty()

                    answer_placeholder.markdown(
                        answer
                    )

                except Exception as exc:

                    status_placeholder.empty()

                    answer = (
                        "### Recruiter Chat Error\n\n"
                        f"**{type(exc).__name__}:** "
                        f"{exc}"
                    )

                    answer_placeholder.error(
                        answer
                    )

            # ----------------------------------------------------
            # SAVE RESPONSE
            # ----------------------------------------------------

            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        # ========================================================
        # CLEAR
        # ========================================================

        if st.session_state.chat:

            st.divider()

            if st.button(
                "Clear conversation"
            ):

                st.session_state.chat = []

                st.rerun()


# ################################################################
# COMPARE CANDIDATES
# ################################################################

with tab_compare:

    st.header(
        "Candidate Comparison"
    )

    st.info(
        "Candidate comparison uses the deterministic matching engine."
    )

    left, right = st.columns(2)

    with left:

        candidate_a_text = st.text_area(
            "Candidate A",
            value=DEFAULT_CANDIDATE,
            height=220,
            key="candidate_a",
        )

    with right:

        candidate_b_text = st.text_area(
            "Candidate B",
            value=(
                "Data analyst with 3 years of experience "
                "using SQL, Python, pandas, statistics and Power BI. "
                "Built dashboards and automated reports."
            ),
            height=220,
            key="candidate_b",
        )

    if st.button(
        "Compare Candidates",
        type="primary",
    ):

        try:

            if st.session_state.job_profile:

                job = (
                    st.session_state.job_profile
                )

            else:

                job = build_job_profile(
                    DEFAULT_JOB
                )

            candidate_a = build_candidate_profile(
                candidate_a_text,
                "A",
            )

            candidate_b = build_candidate_profile(
                candidate_b_text,
                "B",
            )

            score_a = score_candidate(
                candidate_a,
                job,
            )

            score_b = score_candidate(
                candidate_b,
                job,
            )

            st.dataframe(
                {
                    "Metric": [
                        "Overall",
                        "Skills",
                        "Technology",
                        "Experience",
                        "Projects",
                        "Required Requirements",
                    ],
                    "Candidate A": [
                        score_a["overall_score"],
                        score_a["skill_score"],
                        score_a["technology_score"],
                        score_a["experience_score"],
                        score_a["project_score"],
                        score_a[
                            "required_requirement_score"
                        ],
                    ],
                    "Candidate B": [
                        score_b["overall_score"],
                        score_b["skill_score"],
                        score_b["technology_score"],
                        score_b["experience_score"],
                        score_b["project_score"],
                        score_b[
                            "required_requirement_score"
                        ],
                    ],
                },
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                "Candidate comparison failed."
            )

            st.exception(exc)
