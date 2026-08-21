"""Small, transparent retrieval layer for the recruiter RAG pipeline."""

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_documents(candidate, job):
    documents = []
    for entity in candidate.get("entities", []):
        for index, snippet in enumerate(entity.get("evidence", [])[:2]):
            documents.append({
                "id": f"candidate-{entity['name']}-{index}",
                "type": "candidate",
                "text": snippet,
                "metadata": {"entity": entity["name"], "category": entity["category"]},
            })

    documents.append({
        "id": "candidate-profile",
        "type": "candidate",
        "text": (
            f"Skills: {', '.join(candidate.get('skills', []))}. "
            f"Technologies: {', '.join(candidate.get('technologies', []))}. "
            f"Programming languages: {', '.join(candidate.get('languages', []))}. "
            f"Experience: {candidate.get('experience_years', 0)} years. "
            f"Projects: {' '.join(candidate.get('projects', []))}. "
            f"Source: {candidate.get('source_text', '')}"
        ),
        "metadata": {"category": "profile"},
    })

    documents.append({
        "id": "job-profile",
        "type": "job",
        "text": (
            f"Role: {job.get('title')}. "
            f"Required: {', '.join(job.get('required', []))}. "
            f"Preferred: {', '.join(job.get('preferred', []))}. "
            f"Experience: {job.get('experience_years', 0)} years. "
            f"Responsibilities: {' '.join(job.get('responsibilities', []))}. "
            f"Source: {job.get('source_text', '')}"
        ),
        "metadata": {"category": "job"},
    })
    return documents


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokens = [document["text"].lower().split() for document in documents]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens and BM25Okapi else None

    def retrieve(self, query, k=6):
        if not self.documents:
            return []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query.lower().split())
            order = scores.argsort()[::-1][:k]
            return [self.documents[index] for index in order]
        query_tokens = set(query.lower().split())
        return sorted(
            self.documents,
            key=lambda document: len(query_tokens & set(document["text"].lower().split())),
            reverse=True,
        )[:k]


class TfidfRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        texts = [document["text"] for document in documents]
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def retrieve(self, query, k=6):
        if not self.documents or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [self.documents[index] for index in order]


class HybridRetriever:
    """BM25 + TF-IDF rank fusion; transparent and dependency-light."""

    def __init__(self, documents):
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.tfidf = TfidfRetriever(documents)

    def retrieve(self, query, k=6):
        if not self.documents:
            return []

        pool_size = min(len(self.documents), max(k * 2, 8))
        bm25_docs = self.bm25.retrieve(query, k=pool_size)
        tfidf_docs = self.tfidf.retrieve(query, k=pool_size)
        bm25_rank = {document["id"]: index for index, document in enumerate(bm25_docs)}
        tfidf_rank = {document["id"]: index for index, document in enumerate(tfidf_docs)}
        candidates = {document["id"]: document for document in bm25_docs + tfidf_docs}

        scored = []
        for document_id, document in candidates.items():
            b = 1.0 / (60 + bm25_rank[document_id]) if document_id in bm25_rank else 0.0
            t = 1.0 / (60 + tfidf_rank[document_id]) if document_id in tfidf_rank else 0.0
            scored.append((b + t, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:k]]
