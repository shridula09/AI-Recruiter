import re

REQUIRED_WORDS = ("required", "must", "mandatory", "essential", "minimum", "need")
PREFERRED_WORDS = ("preferred", "nice to have", "desirable", "bonus", "plus")


def classify_sentence(sentence):
    low = sentence.lower()
    if any(word in low for word in REQUIRED_WORDS):
        return "required"
    if any(word in low for word in PREFERRED_WORDS):
        return "preferred"
    return "responsibility"


def extract_requirement_sentences(text):
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    result = {"required": [], "preferred": [], "responsibility": [], "experience": []}
    for sentence in sentences:
        category = classify_sentence(sentence)
        result[category].append(sentence)
        if re.search(r"\d+(?:\.\d+)?\s*\+?\s*(?:years?|yrs?)", sentence.lower()):
            result["experience"].append(sentence)
    return result
