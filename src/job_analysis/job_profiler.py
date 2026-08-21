import re

from src.preprocessing.text_cleaner import clean_text
from src.extraction.extractor import extract_entities
from src.job_analysis.requirement_extractor import extract_requirement_sentences

REQUIREMENT_MARKER_WORDS = {
    "required", "preferred", "must", "mandatory", "essential", "minimum", "need",
    "nice to have", "desirable", "bonus", "plus",
}


def _is_requirement_marker(name):
    normalized = re.sub(r"\s+", " ", str(name).strip().lower())
    return normalized in REQUIREMENT_MARKER_WORDS


def build_job_profile(text, job_id=None, title=None):
    text = clean_text(text)
    entities = extract_entities(text)
    classification = extract_requirement_sentences(text)
    required, preferred = [], []

    for entity in entities:
        name = entity.get("name", "").strip()
        if not name or _is_requirement_marker(name):
            continue
        evidence = " ".join(entity.get("evidence", [])).lower()
        if any(x in evidence for x in ["required", "must", "mandatory", "essential", "minimum", "need"]):
            required.append(name)
        elif any(x in evidence for x in ["preferred", "nice to have", "desirable", "bonus", "plus"]):
            preferred.append(name)

    low = text.lower()
    for label, target in [("required", required), ("preferred", preferred)]:
        matches = re.findall(rf"{label}\s+skills?\s*:\s*([^\.]+)", low)
        for section in matches:
            section_terms = [
                item.strip() for item in re.split(r",|;|\band\b", section, flags=re.I) if item.strip()
            ]
            for entity in entities:
                name = entity.get("name", "").strip()
                if not name or _is_requirement_marker(name):
                    continue
                if any(name.lower() == term.lower() or name.lower() in term.lower() for term in section_terms):
                    target.append(name)

    exp_matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", low)
    exp_years = max([float(x) for x in exp_matches], default=0.0)

    return {
        "job_id": job_id,
        "title": title or (text.split(".")[0][:100] if text else "Unknown Role"),
        "required": sorted({x.strip().lower() for x in required if x.strip() and not _is_requirement_marker(x)}),
        "preferred": sorted({x.strip().lower() for x in preferred if x.strip() and not _is_requirement_marker(x)}),
        "experience_years": exp_years,
        "responsibilities": classification["responsibility"],
        "requirement_sentences": classification,
        "entities": entities,
        "source_text": text,
    }
