def rerank(query, documents):
    """Deterministic lexical reranking of the small retrieved candidate set."""
    query_tokens = set(query.lower().split())

    def score(document):
        tokens = set(document["text"].lower().split())
        return len(query_tokens & tokens)

    return sorted(documents, key=score, reverse=True)
