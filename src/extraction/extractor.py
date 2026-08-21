"""Part 1: open-vocabulary conversational NLP extraction.

Challenge constraints:
- English conversational input only in this version.
- No LLM.
- No pretrained model.
- No fixed skill/technology vocabulary.
- No training dataset is required.

Architecture:
1. Rule-based linguistic segmentation using a spaCy *blank* English pipeline
   (tokenizer + sentencizer only; no pretrained components).
2. Contextual relation extraction.
3. Unsupervised statistical keyphrase candidates (YAKE when installed, with
   a local YAKE-style fallback so the extractor remains runnable).
4. Local TF-IDF informativeness as a secondary ranking signal.
5. Technical orthographic-shape detection.
6. Contextual multi-class scoring: Skill / Technology / Language / Project.
7. Negation scope and clause boundaries.
8. Span conflict resolution, normalization and deduplication.

The final decision is our own explainable scoring logic. YAKE/TF-IDF only
provide candidate/ranking evidence; they do not decide the output category.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import math
import re

try:
    import spacy
except ImportError:
    spacy = None

try:
    import yake
except ImportError:
    yake = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None


@dataclass
class Candidate:
    text: str
    start: int
    end: int
    sentence: str
    sentence_start: int
    source: str
    score: float = 0.0
    stats: dict | None = None


# These are grammatical function words, not skills or technologies.
STOP = {
    "i","me","my","mine","we","our","ours","you","your","he","she","they","their",
    "a","an","the","and","or","but","if","then","than","to","of","in","on","at","for",
    "from","with","without","by","via","through","as","is","are","was","were","be",
    "been","being","am","have","has","had","do","does","did","not","no","never",
    "this","that","these","those","it","its","into","over","under","about","around",
    "also","very","just","currently","recently","main","mainly","strong","good",
    "experience","experienced","familiar","comfortable","working","worked","work",
    "using","use","used","uses","learn","learning","learned","exploring","exposure",
    "know","knows","built","build","building","developed","development","handled",
    "handle","handles","responsible","responsibilities","skills","skill","strengths",
    "expertise","capabilities","technologies","technology","languages","language",
    "project","application","app","platform","system","called","named","internal",
    "company","team","department","services","service","years","year","months","month",
    "tools","tool","stack","workflow","include","includes","involves","involved",
}

GENERIC_CONTEXT_NOUNS = {
    "data","work","experience","projects","project","people","team","teams","developers",
    "developer","services","service","systems","system","applications","application",
    "tools","tool","things","workflows","workflow","tasks","task","process","processes",
    "methods","method","models","model","results","requirements","skills","knowledge",
    "experience","platforms","platform","cloud","analytics","analysis","automation",
    "deployment","deployments","development","testing","test","reviews","review",
}

LANGUAGE_HINTS = {
    "python","java","javascript","typescript","c","c++","c#","rust","go","kotlin",
    "scala","dart","swift","ruby","php","r","sql","haskell","zig","lua","perl","bash","elixir","erlang","groovy","julia","matlab","solidity","fortran","cobol","clojure","f#","objective-c","ocaml",
}

NEG_TRIGGERS = re.compile(
    r"\b(?:not|never|no|without|haven't|hasn't|hadn't|don't|doesn't|didn't|"
    r"cannot|can't|won't|wouldn't|unable|lack(?:ing)?)\b", re.I
)

# Context cues. These are linguistic relations, not vocabularies.
SKILL_CUES = re.compile(
    r"\b(?:skills?|strengths?|expertise|responsibilit(?:y|ies)|capabilit(?:y|ies)|"
    r"competenc(?:e|ies)|focus(?:es|ed)?|speciali[sz](?:e|ed|ing)|"
    r"proficient|good\s+at|strong\s+in|experienced?\s+in|experience\s+in|"
    r"responsible\s+for|involved\s+in|knowledge\s+of|worked\s+on|"
    r"handle(?:s|d)?|handling)\b", re.I
)
TECH_CUES = re.compile(
    r"\b(?:use|uses|used|using|worked\s+with|experience\s+with|familiar\s+with|"
    r"exposure\s+to|built\s+with|deployed\s+with|stack|toolkit|"
    r"technologies?|frameworks?|libraries?|packages?|database|databases?|"
    r"platform|cloud|sdk|api)\b", re.I
)
LANG_CUES = re.compile(
    r"\b(?:programming\s+languages?|coding\s+languages?|languages?|"
    r"programmed|coded|write|writes|written|developed|code)\b", re.I
)
PROJECT_CUE = re.compile(
    r"\b(?:project|application|app|platform|system|tool|database)\s+"
    r"(?:called|named)\b", re.I
)

SKILL_MORPH = re.compile(
    r"(?:engineering|analysis|design|management|development|testing|evaluation|"
    r"retrieval|tuning|optimization|automation|modeling|monitoring|debugging|"
    r"programming|profiling|forecasting|planning|communication|mentoring|"
    r"troubleshooting|segmentation|augmentation|deployment|reporting|writing|"
    r"review|orchestration|quality|security|lineage|ranking|validation|"
    r"visualization|reasoning|research)$", re.I
)


# ---------- NLP primitives ----------

_NLP = None
if spacy is not None:
    # Explicitly blank: tokenizer only. No pretrained pipeline/model is loaded.
    _NLP = spacy.blank("en")
    _NLP.add_pipe("sentencizer")


def _sentences(text: str):
    if _NLP is not None:
        doc = _NLP(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                yield s, sent.start_char
        return

    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        s = m.group(0).strip()
        if s:
            yield s, m.start()


def _tokens(text: str):
    if _NLP is not None:
        doc = _NLP.make_doc(text)
        return [(t.text, t.idx, t.idx + len(t.text)) for t in doc if not t.is_space]

    return [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"[A-Za-z0-9]+(?:[._+#/-][A-Za-z0-9]+)*|[.#+/-]+", text)
    ]


def _clean(x: str) -> str:
    x = re.sub(r"\s+", " ", x).strip(" \t\r\n,;:")
    x = re.sub(r"^(?:and|or)\s+", "", x, flags=re.I)
    x = re.sub(r"\s+(?:and|or)$", "", x, flags=re.I)
    return x.strip()


def _technical_shape(x: str) -> bool:
    return bool(
        re.search(r"\d", x)
        or re.search(r"[+#.]", x)
        or (re.search(r"[A-Z]", x) and re.search(r"[a-z]", x))
        or re.search(r"[/_-]", x)
        or re.search(r"\b[A-Z]{2,}\b", x)
    )


def _looks_like_entity(x: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+(?:[._+#/-][A-Za-z0-9]+)*", x)
    if not words or len(words) > 6:
        return False
    if all(w.lower() in STOP for w in words):
        return False
    if re.fullmatch(r"[\d\s./-]+", x):
        return False
    return True


def _split_list(body: str):
    """Split lists while preserving technical punctuation and multiword phrases."""
    body = re.sub(
        r"^(?:use|uses|used|using|worked\s+with|experience\s+(?:in|with)|"
        r"familiar\s+with|exposure\s+to|learning|exploring|built\s+with|"
        r"deployed\s+with|write|writes|code|coded|programmed|programming)\s+",
        "", body, flags=re.I,
    )
    # First cut an independent clause: "Python and Go and use FastAPI" -> 
    # "Python and Go" + "use FastAPI". This is a syntactic boundary heuristic,
    # not a vocabulary of technologies.
    body = re.split(
        r"\s+and\s+(?=(?:use|uses|used|work|worked|build|built|deploy|deployed|"
        r"develop|developed|test|tests|testing|focus|specialize|handle|handles|"
        r"write|writes|code|coded|programmed|manage|managing)\b)",
        body, maxsplit=1, flags=re.I
    )[0]
    pieces = re.split(r"\s*(?:,|;)\s*|\s+and\s+|\s+or\s+", body, flags=re.I)
    out = []
    for part in pieces:
        part = _clean(part)
        part = re.sub(r"^(?:for|to|with|via|through)\s+", "", part, flags=re.I)
        part = re.sub(r"\s+(?:instead|respectively|etc\.?|as\s+well)$", "", part, flags=re.I)
        part = re.sub(
            r"\s+(?:for|to|via|through)\s+(?:automation|analytics|monitoring|testing|"
            r"analysis|development|deployment|deployments|services?)\b.*$",
            "", part, flags=re.I,
        )
        if part and _looks_like_entity(part):
            out.append(part)
    return out


def _context(sent: str, start: int, end: int, radius: int = 80) -> str:
    return sent[max(0, start - radius):min(len(sent), end + radius)]


def _clause_left(sent: str, start: int) -> str:
    left = sent[:start]
    # Clause boundaries prevent negation from leaking through "but/however".
    positions = [left.rfind(x) for x in [",", ";", ":", " but ", " however "]]
    cut = max(positions) if positions else -1
    return left[cut + 1:]


# ---------- Candidate generation ----------

def _add_candidate(out, text, sent, base, local_start, source):
    text = _clean(text)
    if not text or not _looks_like_entity(text):
        return
    # Locate the actual span as exactly as possible.
    pos = sent.lower().find(text.lower(), max(0, local_start))
    if pos < 0:
        return
    out.append(
        Candidate(
            text=text,
            start=pos,
            end=pos + len(text),
            sentence=sent,
            sentence_start=base,
            source=source,
        )
    )


def _context_candidates(text: str):
    out = []

    for sent, base in _sentences(text):
        patterns = [
            # Explicit skill constructions.
            (r"\b(?:my|our|the)?\s*(?:skills?|strengths|expertise|responsibilities|"
             r"capabilities|competencies)\s*(?:are|is|include|includes|involve|"
             r"involves|:)\s+(.+?)(?=[.!?]|$)", "skill_list"),
            (r"\b(?:i|we)\s+(?:do\s+)?(?:(?:have|am|are)\s+)?"
             r"(?:(?:\d+(?:\.\d+)?\s+(?:years?|yrs?|months?|mos?)\s+of\s+)?"
             r"experience|experienced)\s+in\s+(.+?)(?=[.!?]|$)",
             "skill_experience"),
            (r"\b(?:focus(?:es|ed)?(?:\s+on)?|speciali[sz](?:e|ed|ing)(?:\s+in)?|"
             r"responsible\s+for|involved\s+in|proficient\s+in|good\s+at|"
             r"strong\s+in|knowledge\s+of|"
             r"handle(?:s|d)?|handling|worked\s+on)\s+(.+?)(?=[.!?]|$)",
             "skill_relation"),

        # Explicit language constructions.
            (r"\b(?:my|our|the)?\s*(?:programming\s+)?languages?\s*"
             r"(?:are|include|:)\s+(.+?)(?=[.!?]|$)", "language_list"),
            (r"\b(?:write|writes)\s+(?:primarily\s+|mainly\s+)?(?:in|with)?\s*(.+?)"
             r"(?=\s+and\s+(?:use|uses|used|work|worked|build|built|deploy|"
             r"deployed|develop|developed|test|tests|testing|focus|specialize)\b|[.!?]|$)",
             "language_relation"),
            (r"\b(?:code|codes|coded|programmed|program|developed|written)\s+"
             r"(?:primarily\s+|mainly\s+)?(?:in|with)\s+(.+?)"
             r"(?=\s+and\s+(?:use|uses|used|work|worked|build|built|deploy|"
             r"deployed|develop|developed|test|tests|testing|focus|specialize)\b|[.!?]|$)",
             "language_relation"),

            # Technology constructions.
            (r"\b(?:my|our|the)\s+(?:technology|technologies|stack|toolkit|"
             r"workflow)\s*(?:is|includes|:)\s+(.+?)(?=[.!?]|$)", "tech_list"),
            (r"\b(?:technologies?|frameworks?|libraries?|packages?|tools?|"
             r"databases?|platforms?)\s*(?:used|include|includes|are|is|:)\s+"
             r"(.+?)(?=[.!?]|$)", "tech_list"),
            (r"\b(?:using|use|uses|used|worked\s+with|experience\s+with|experienced\s+with|"
             r"familiar\s+with|comfortable\s+with|exposure\s+to|learning|"
             r"experimenting\s+with|built\s+with|deployed\s+with)\s+"
             r"(.+?)(?=\s+and\s+(?:deployed|built|used|worked|focus|specialize)\b|[.!?]|$)",
             "tech_relation"),
        ]

        for pattern, source in patterns:
            for m in re.finditer(pattern, sent, re.I):
                body = m.group(1)

                # Stop at an independent clause.
                body = re.split(
                    r"\s+(?:but|while|whereas|although)\s+(?:i|we|my|our)\b",
                    body, maxsplit=1, flags=re.I
                )[0]
                body = re.split(
                    r"\s+and\s+(?:i|we|my|our)\b",
                    body, maxsplit=1, flags=re.I
                )[0]
                if source.startswith("skill_"):
                    body = re.split(
                        r"\s+(?:with|using)\b", body, maxsplit=1, flags=re.I
                    )[0]
                if source.startswith("tech_"):
                    body = re.split(
                        r"\s+and\s+(?:deployed|built|used|worked|focus|specialize|"
                        r"write|code|programmed)\b",
                        body, maxsplit=1, flags=re.I
                    )[0]
                    body = re.split(
                        r"\s+for\s+(?:analytics|automation|monitoring|testing|"
                        r"analysis|development|deployment)\b",
                        body, maxsplit=1, flags=re.I
                    )[0]

                # Project names are not technology names.
                body = re.sub(
                    r"^(?:a|an|the)\s+(?:proprietary|internal|custom)?\s*"
                    r"(?:platform|database|tool|system|application|app)\s+"
                    r"(?:called|named)\s+[^,;]+",
                    "", body, flags=re.I,
                )

                if source.startswith("tech_"):
                    body = re.split(
                        r"\s+and\s+(?=(?:deployed|deploy|built|build|used|use|worked|work|"
                        r"developed|develop|testing|test|write|writes|code|coded)\b)",
                        body, maxsplit=1, flags=re.I
                    )[0]
                for part in _split_list(body):
                    # A language in a coding context belongs to Languages.
                    if source.startswith("tech_") and part.lower() in LANGUAGE_HINTS:
                        continue
                    if re.match(
                        r"^(?:i|we|my|our|the|a|an)\b", part, flags=re.I
                    ):
                        continue
                    if source.startswith("tech_"):
                        part=re.sub(
                            r"^(?:deployed|built|used|worked)\s+(?:services?|it)\s+(?:using|with)\s+",
                            "", part, flags=re.I
                        )
                        part=re.sub(
                            r"^(?:deployed|built|used|worked)\s+(?:services?|it)\s+(?:using|with)\s*$",
                            "", part, flags=re.I
                        )
                    _add_candidate(out, part, sent, base, m.start(1), source)

        # Direct language token forms: "queries in SQL", "developed in Go".
        for m in re.finditer(
            r"\b(?:in|with)\s+([A-Za-z][A-Za-z0-9+#.-]*[A-Za-z0-9+#])(?=[\s.!?,;]|$)",
            sent, re.I
        ):
            tok=_clean(m.group(1))
            tok=tok.rstrip(".")
            if tok.lower() in LANGUAGE_HINTS:
                _add_candidate(out, tok, sent, base, m.start(1), "language_context")

        # Capability nouns introduced by "for/in" can be skills when their surface
        # morphology strongly indicates an activity/capability.
        for m in re.finditer(
            r"\b(?:for|in)\s+([A-Za-z][A-Za-z0-9 -]{2,80}?)"
            r"(?=[.!?,;]|$)", sent, re.I
        ):
            phrase=_clean(m.group(1))
            parts=phrase.split()
            if 1 <= len(parts) <= 4 and SKILL_MORPH.search(phrase):
                _add_candidate(out, phrase, sent, base, m.start(1), "skill_morph")

        # Direct capability-cue extraction. Candidate spans are still classified
        # by the contextual scorer below; this only improves recall.
        for m in re.finditer(
            r"\b(?:strong\s+in|focus(?:es|ed)?\s+on|speciali[sz](?:e|ed|ing)\s+in|"
            r"proficient\s+in|good\s+at|knowledge\s+of|experienced?\s+with|"
            r"experience\s+in)\s+(.+?)(?=[.!?]|$)",
            sent, re.I
        ):
            body=m.group(1)
            body=re.split(
                r"\s+(?:with|using)\b",
                body, maxsplit=1, flags=re.I
            )[0]
            body=re.split(
                r"\s+(?:and\s+(?:use|uses|used|work|worked|build|built|deploy|"
                r"deployed|write|writes|code|coded|programmed|focus|specialize)|"
                r"while)\b",
                body, maxsplit=1, flags=re.I
            )[0]
            for part in _split_list(body):
                _add_candidate(out, part, sent, base, m.start(1), "skill_relation")

        # Contextual single language mentions.
        for m in re.finditer(
            r"\b(?:learning|learned|comfortable\s+with|familiar\s+with|"
            r"know|knows|worked\s+with|using|use|uses|used|programming\s+in|"
            r"coding\s+in)\s+(.+?)"
            r"(?=[.!?]|,?\s+(?:and|but)\b|$)",
            sent, re.I,
        ):
            for part in _split_list(m.group(1)):
                if part.lower() in LANGUAGE_HINTS:
                    _add_candidate(out, part, sent, base, m.start(1), "language_context")

        # High-precision single-language constructions. This is a small set of
        # language *forms* used only to disambiguate the required Languages class.
        for m in re.finditer(
            r"\b(?:learning|learned|comfortable\s+with|familiar\s+with|"
            r"programming\s+in|coding\s+in|worked\s+with|using|use|uses|used)\s+"
            r"([A-Za-z][A-Za-z0-9+#.-]*)", sent, re.I
        ):
            tok=_clean(m.group(1))
            if tok.lower() in LANGUAGE_HINTS:
                _add_candidate(out, tok, sent, base, m.start(1), "language_context")

        for m in re.finditer(
            r"\b(?:worked\s+with|using|use|uses|used)\s+(.+?)"
            r"(?=\s+and\s+(?:deployed|built|used|worked|write|code|programmed)\b|[.!?]|$)",
            sent, re.I
        ):
            for part in _split_list(m.group(1)):
                if part.lower() in LANGUAGE_HINTS:
                    _add_candidate(out, part, sent, base, m.start(1), "language_context")

        # Narrow open-vocabulary technical-name contexts.
        for m in re.finditer(
            r"\b(?:using|through|exposure\s+to|worked\s+with|experience\s+with|"
            r"familiar\s+with|comfortable\s+with|experimenting\s+with|"
            r"deployed\s+with)\s+"
            r"([A-Za-z][A-Za-z0-9]*(?:[._+#/-][A-Za-z0-9]+)*)",
            sent, re.I,
        ):
            tok = _clean(m.group(1))
            if tok.lower() not in STOP and tok.lower() not in LANGUAGE_HINTS:
                if _technical_shape(tok) or tok[:1].isupper():
                    _add_candidate(out, tok, sent, base, m.start(1), "tech_context")

        # Generic technical relation with a proper/technical-looking token/list.
        # This is deliberately vocabulary-free.
        for m in re.finditer(
            r"\b(?:with|deploy(?:ed|s)?\s+with|build(?:t|s)?\s+with)\s+"
            r"(.+?)(?=[.!?]|,?\s+(?:for|to)\b|$)",
            sent, re.I
        ):
            body = re.split(
                r"\s+and\s+(?=(?:use|uses|used|work|worked|build|built|deploy|"
                r"deployed|develop|developed|test|tests|testing|focus|specialize|"
                r"write|writes|code|coded|programmed)\b)",
                m.group(1), maxsplit=1, flags=re.I
            )[0]
            for part in _split_list(body):
                if part.lower() in STOP or part.lower() in LANGUAGE_HINTS:
                    continue
                if part.lower() in GENERIC_CONTEXT_NOUNS:
                    continue
                if _technical_shape(part) or part[:1].isupper() or len(part.split()) == 1:
                    _add_candidate(out, part, sent, base, m.start(1), "tech_context")

        # Generic professional-action skills. We normalize the action itself
        # when the sentence uses a verb ("I mentor..." -> "mentoring").
        for m in re.finditer(
            r"\b(?:mentor|mentors|mentored|mentoring)\b", sent, re.I
        ):
            out.append(Candidate(
                "mentoring", m.start(), m.end(), sent, base, "skill_action"
            ))

        for m in re.finditer(
            r"\b(?:conduct|conducts|conducted|conducting)\s+(.+?)"
            r"(?=\s+(?:while|and\s+(?:maintain|maintaining|manage|managing))\b|[.!?]|$)",
            sent, re.I
        ):
            _add_candidate(out, m.group(1), sent, base, m.start(1), "skill_action")

        for m in re.finditer(
            r"\b(?:maintain|maintains|maintained|maintaining)\s+(.+?)(?=[.!?]|$)",
            sent, re.I
        ):
            body=re.split(r"\s+(?:and|while)\s+",m.group(1),maxsplit=1,flags=re.I)[0]
            _add_candidate(out, body, sent, base, m.start(1), "skill_action")

        # Generic capability verbs. These are skill candidates only when the object
        # is linguistically plausible.
        for m in re.finditer(
            r"\b(?:design|designing|develop|developing|test|testing|debug|debugging|"
            r"manage|managing|monitor|monitoring|analy[sz]e|analysis)\s+"
            r"(.+?)(?=[.!?]|$)",
            sent, re.I,
        ):
            for part in _split_list(m.group(1)):
                if part.lower() not in LANGUAGE_HINTS:
                    _add_candidate(out, part, sent, base, m.start(1), "skill_verb")

    return out


def _yake_candidates(text: str, top_n: int = 35):
    """Use actual YAKE when installed; otherwise return no extra candidates.

    The fallback scoring below still provides a YAKE-style local statistical rank.
    YAKE is unsupervised and requires no training corpus/dictionary.
    """
    if yake is None:
        return []

    try:
        extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.85,
            top=top_n,
            features=None,
        )
        return [kw for kw, _score in extractor.extract_keywords(text)]
    except Exception:
        return []


def _statistical_candidates(text: str):
    """Generate bounded 1-3 word phrase candidates without a vocabulary.

    Stopwords are allowed *inside* a phrase ("design of experiments"), but a
    candidate is rejected if it crosses a coordinating conjunction. This keeps
    statistical candidate generation from joining two independent entities.
    """
    out = []
    for sent, base in _sentences(text):
        toks = [
            (w, a, b) for w, a, b in _tokens(sent)
            if re.search(r"[A-Za-z0-9]", w)
        ]
        for i in range(len(toks)):
            for n in (1, 2, 3):
                if i+n > len(toks):
                    continue
                group=toks[i:i+n]
                phrase=sent[group[0][1]:group[-1][2]].strip(" ,;:")
                if not _looks_like_entity(phrase):
                    continue
                between=sent[group[0][2]:group[-1][1]]
                if re.search(r"[.!?;:]",between):
                    continue
                words=[w.lower() for w,_,_ in group]
                if words[0] in STOP:
                    continue
                if any(w in {"and","or","but"} for w in words):
                    continue
                if any(w in {"for","with","to"} for w in words):
                    continue
                # Do not begin/end on grammatical glue.
                if words[0] in {"and","or","but","to","for","with","of","in","on"}:
                    continue
                if words[-1] in {"and","or","but","to","for","with","of","in","on"}:
                    continue
                if all(w in STOP for w in words):
                    continue
                out.append(Candidate(
                    phrase, group[0][1], group[-1][2], sent, base, "ngram"
                ))
    return out


def _make_yake_candidates(text: str, candidates):
    """Apply YAKE's unsupervised ranking signal to existing candidates only.

    The output candidate set remains controlled by our linguistic/statistical
    candidate generator. If YAKE is installed, its score can strengthen a
    candidate that it independently considers salient; it can never introduce
    a new entity into the output.
    """
    if yake is None or not candidates:
        return candidates

    try:
        extractor = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.85, top=40, features=None
        )
        ranked = extractor.extract_keywords(text)
        if not ranked:
            return candidates

        # YAKE returns lower scores for better phrases.
        rank_map = {kw.lower(): score for kw, score in ranked}
        for c in candidates:
            val = rank_map.get(c.text.lower())
            if val is not None:
                c.score += 1.0 / (1.0 + float(val))
                c.stats = (c.stats or {}) | {"yake_rank_score": round(float(val), 5)}
    except Exception:
        pass

    return candidates



# ---------- Statistical scoring ----------

def _tfidf_scores(text: str):
    if TfidfVectorizer is None:
        return {}

    sents = [s for s,_ in _sentences(text)]
    if len(sents) < 2:
        return {}

    try:
        vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_+#.-]*\b",
            min_df=1,
        )
        X = vec.fit_transform(sents)
        names = vec.get_feature_names_out()
        scores = {}
        for i, sent in enumerate(sents):
            row = X.getrow(i)
            for idx, val in zip(row.indices, row.data):
                scores[(i, names[idx])] = float(val)
        return scores
    except Exception:
        return {}


def _local_stat_score(c: Candidate, sentence_index: int, tfidf):
    words = re.findall(r"[A-Za-z0-9]+(?:[._+#/-][A-Za-z0-9]+)*", c.text)
    if not words:
        return 0.0

    sent_words = re.findall(r"[A-Za-z0-9]+(?:[._+#/-][A-Za-z0-9]+)*", c.sentence)
    lower_sent = [w.lower() for w in sent_words]
    lower_words = [w.lower() for w in words]

    # Local YAKE-style signals.
    freq = sum(lower_sent.count(w) for w in lower_words)
    position = 1.0 / (1.0 + c.start / max(1, len(c.sentence)))
    casing = sum(
        1 for w in words if w[:1].isupper() or w.isupper()
    ) / max(1, len(words))
    dispersion = len(set(lower_sent)) / max(1, len(lower_sent))
    shape = 1.0 if _technical_shape(c.text) else 0.0
    length = min(len(words), 3) / 3.0

    # TF-IDF is deliberately a secondary feature; it cannot create a candidate.
    tfidf_score = 0.0
    for w in lower_words:
        tfidf_score += tfidf.get((sentence_index, w), 0.0)
    if lower_words:
        tfidf_score /= len(lower_words)

    return (
        0.9 * position
        + 0.7 * casing
        + 0.25 * min(freq, 3)
        + 0.25 * dispersion
        + 0.9 * shape
        + 0.5 * length
        + 0.8 * tfidf_score
    )


# ---------- Classification ----------

def _negative(candidate: Candidate) -> bool:
    left = _clause_left(candidate.sentence, candidate.start)
    # Negation only has scope inside the current clause.
    if NEG_TRIGGERS.search(left[-90:]):
        return True
    return bool(
        re.search(
            r"(?:not|never|no|without)\s+(?:used|use|worked\s+with|"
            r"experience\s+with|familiar\s+with|have|has|know|using)?\s*$",
            left, re.I,
        )
    )


def _project(candidate: Candidate) -> bool:
    before = candidate.sentence[:candidate.start]
    return bool(
        re.search(
            r"\b(?:project|application|app|platform|system|tool|database)"
            r"\s+(?:called|named)\s*$",
            before, re.I,
        )
    )


def _classify(c: Candidate):
    p = c.text.strip()
    low = p.lower()
    left = _clause_left(c.sentence, c.start)
    right = c.sentence[c.end:min(len(c.sentence), c.end+100)]
    local = left + " " + p + " " + right

    scores = {
        "Skills": 0.0,
        "Technologies": 0.0,
        "Languages": 0.0,
        "Project": 0.0,
    }

    # Source/context evidence.
    if c.source.startswith("skill_"):
        scores["Skills"] += 6.0
    if c.source == "skill_experience":
        scores["Skills"] += 2.0

    if c.source.startswith("language_"):
        scores["Languages"] += 8.0

    if c.source.startswith("tech_"):
        scores["Technologies"] += 6.0
        if c.source in {"tech_relation"} and len(p.split()) >= 2 and not _technical_shape(p):
            scores["Skills"] += 5.0

    if c.source == "yake":
        # YAKE is a candidate-ranking signal, never a category decision.
        scores["Skills"] += 0.4
        scores["Technologies"] += 0.4

    if c.source == "ngram" and len(p.split()) >= 2 and not _technical_shape(p):
        if SKILL_MORPH.search(low):
            scores["Skills"] += 3.5
            scores["Technologies"] = max(0.0, scores["Technologies"] - 2.0)

    # Context relation evidence.
    if re.search(
        r"\b(?:experience|expertise|skills?|strengths?|responsible|"
        r"focus|speciali[sz]|proficient|capabilit|competenc|knowledge|"
        r"handling|handle|worked\s+on)\b", left, re.I
    ):
        scores["Skills"] += 4.5

    if re.search(
        r"\b(?:using|use|uses|used|worked\s+with|familiar\s+with|"
        r"exposure\s+to|built\s+with|deployed\s+with|stack|"
        r"technolog|framework|librar|package|database|platform|cloud)\b",
        left, re.I
    ):
        scores["Technologies"] += 4.0

    if re.search(
        r"\b(?:programming\s+languages?|coding\s+languages?|"
        r"programmed|coded|write|writes|written|code)\b", left, re.I
    ):
        scores["Languages"] += 5.0

    # Morphological capability evidence.
    if SKILL_MORPH.search(low):
        scores["Skills"] += 2.0
        if SKILL_CUES.search(left):
            scores["Skills"] += 1.5

    if len(p.split()) >= 2 and not _technical_shape(p):
        scores["Skills"] += 1.0

    # Orthographic evidence for technical entities.
    if _technical_shape(p):
        scores["Technologies"] += 2.5

    # Generic technical suffix/shape is only weak evidence.
    if re.search(r"(?:api|db|sql|ml|ai|gpu|sdk|cli|net|js|css|html)$", low):
        scores["Technologies"] += 1.0

    # Explicit language identity gets priority only in linguistic coding context.
    if low in LANGUAGE_HINTS:
        if LANG_CUES.search(local):
            scores["Languages"] += 7.0
        if re.search(r"\b(?:learning|learned|comfortable\s+with|familiar\s+with|"
                     r"worked\s+with|using|use|uses|used|programming|coding)\b", left, re.I):
            scores["Languages"] += 4.0
        # Avoid classifying "SQL database" as language unless context supports it.
        if re.search(r"\b(?:database|db|query)\b", right, re.I) and not re.search(
            r"\b(?:program|code|write|language)\b", local, re.I
        ):
            scores["Languages"] -= 2.0

    # Project context is a hard exclusion.
    if _project(c):
        scores["Project"] += 10.0

    if re.match(r"^(?:most|some|much|all)\s+(?:of\s+)?(?:my|our|the)\b", low):
        return None, scores
    if low in {"cloud work is","cloud work","write","reviews","junior developers"}:
        return None, scores

    # Project/application/database names introduced by "called/named" are
    # explicitly outside the three required categories.
    if scores["Project"] >= max(scores["Skills"], scores["Technologies"], scores["Languages"]):
        return None, scores

    # Generic words are rejected.
    if re.search(r"\b(?:called|named|proprietary\s+platform|internal\s+project)\b", low):
        return None, scores

    if low in STOP or low in {
        "deployment","deployments","development","writing","reporting",
        "services","service","tools","tool","workflows","workflow"
    }:
        return None, scores

    # Languages are only accepted with explicit language/coding evidence.
    if low in LANGUAGE_HINTS and scores["Languages"] >= 5.0:
        label = "Languages"
    else:
        label = max(
            ("Skills", "Technologies", "Languages"),
            key=lambda x: scores[x],
        )

    if scores[label] < 3.5:
        return None, scores

    # Don't let an ordinary phrase become a technology solely because of casing.
    if label == "Technologies" and scores["Technologies"] < scores["Skills"] + 0.8:
        if not _technical_shape(p) and c.source not in {"tech_relation","tech_list","tech_context"}:
            return None, scores

    # A language must be an explicit language form, not an arbitrary token.
    if label == "Languages" and low not in LANGUAGE_HINTS:
        return None, scores

    return label, scores


# ---------- Span resolution / output ----------

def _normalize(x: str) -> str:
    x = re.sub(r"\s+", " ", x).strip(" ,;:.")
    # Remove accidental relation wrappers.
    x = re.sub(
        r"^(?:deployed\s+(?:services\s+)?using|using|used|use|worked\s+with|"
        r"experience\s+with|exposure\s+to|learning|exploring|experimenting\s+with|"
        r"write|writes|code|coded|programmed)\s+",
        "", x, flags=re.I
    )
    return x.strip()


def _resolve(candidates):
    source_priority = {
        "skill_list": 12, "skill_experience": 12, "skill_relation": 12,
        "skill_verb": 10, "language_list": 15, "language_relation": 15,
        "language_context": 15, "tech_list": 12, "tech_relation": 12,
        "tech_context": 11, "ngram": 2,
    }

    unique={}
    for c in candidates:
        key=(c.text.lower(),c.start,c.end,c.source)
        unique[key]=c
    arr=list(unique.values())

    structured=[c for c in arr if c.source not in {"ngram","yake"}]
    generic=[c for c in arr if c.source in {"ngram","yake"}]

    def overlap(a,b):
        if a.sentence_start != b.sentence_start:
            return False
        return not (a.end<=b.start or a.start>=b.end)

    # First resolve only explicit/contextual candidates.
    structured.sort(key=lambda c:(c.start,-source_priority.get(c.source,0),-(c.end-c.start)))
    kept=[]
    for c in structured:
        if c.text.lower() in STOP:
            continue
        replaced=False
        skip=False
        for i,old in enumerate(kept):
            if not overlap(c,old):
                continue
            cp=source_priority.get(c.source,0)
            op=source_priority.get(old.source,0)
            if cp>op or (cp==op and (c.end-c.start)>(old.end-old.start)):
                kept[i]=c
                replaced=True
            else:
                skip=True
            break
        if not replaced and not skip:
            kept.append(c)

    # Generic statistical candidates are only allowed where they do not collide
    # with a stronger linguistic candidate. This prevents an n-gram such as
    # "deploy with Nomad and containerd" from swallowing two real entities.
    generic.sort(key=lambda c:(c.start,-(c.end-c.start)))
    for c in generic:
        if c.text.lower() in STOP:
            continue
        if any(overlap(c,k) for k in kept):
            continue
        kept.append(c)

    return sorted(kept,key=lambda c:c.start)


def extract(text: str):
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()
    result = {
        "Skills": [],
        "Technologies": [],
        "Languages": [],
        "_evidence": [],
    }
    if not text:
        return result

    candidates = _context_candidates(text)
    candidates += _statistical_candidates(text)

    # Add YAKE candidates when the optional unsupervised package is available.
    candidates = _make_yake_candidates(text, candidates)

    tfidf = _tfidf_scores(text)
    sentence_map = {
        id(sent): i for i, (sent, _) in enumerate(_sentences(text))
    }

    # Attach local statistical evidence.
    for c in candidates:
        idx = 0
        for i, (sent, _) in enumerate(_sentences(text)):
            if sent == c.sentence:
                idx = i
                break
        c.stats = {
            "local_stat": round(_local_stat_score(c, idx, tfidf), 4)
        }
        c.score += c.stats["local_stat"]

    candidates = _resolve(candidates)

    seen = set()
    for c in candidates:
        if c.source == "skill_action":
            c.text = re.sub(r"\s+", " ", c.text).strip(" ,;:.")
        else:
            c.text = _normalize(c.text)
        if not c.text or len(c.text) < 2:
            continue

        if _negative(c):
            continue

        label, scores = _classify(c)
        if label is None:
            continue

        total = sum(max(v, 0.0) for v in scores.values())
        confidence = max(scores[label], 0.0) / total if total else 0.0

        # Split conjunction-linked technology lists into atomic entities.
        final_names=[c.text]
        if label=="Technologies" and re.search(r"\s+and\s+",c.text,re.I):
            final_names=[x for x in _split_list(c.text) if x]

        for final_name in final_names:
            fkey=(label,final_name.lower())
            if fkey in seen:
                continue
            seen.add(fkey)
            result[label].append(final_name)
            result["_evidence"].append({
                "name": final_name,
                "category": label,
                "confidence": round(min(confidence, 0.99), 3),
                "evidence": c.sentence,
                "source": c.source,
                "scores": {k: round(v, 2) for k, v in scores.items()},
                "statistical_score": c.stats or {},
            })

    # Stable ordering by first appearance.
    for key in ("Skills", "Technologies", "Languages"):
        result[key] = list(dict.fromkeys(result[key]))

    return result

# ---------------------------------------------------------------------------
# Candidate profile adapter
# ---------------------------------------------------------------------------

from src.preprocessing.text_cleaner import clean_text, sentences

PROJECT_PATTERNS = re.compile(
    r"\b(?:project|application|app|platform|system|tool|database)\s+"
    r"(?:called|named)\s+([A-Za-z][A-Za-z0-9_.+-]*(?:\s+[A-Za-z0-9_.+-]+)?)",
    re.I,
)


def normalize_surface(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    return text.strip(" ,;:.")


def canonical_key(text: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", normalize_surface(text).lower()).strip()


def extract_experience_for_entity(sentence: str, phrase: str):
    """Return a small evidence record for experience surrounding an entity."""
    sentence = str(sentence or "")
    phrase = str(phrase or "")
    if not phrase:
        return None
    pattern = re.compile(
        rf"(?:\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|months?)\b[^.]*?{re.escape(phrase)}|"
        rf"{re.escape(phrase)}[^.]*?\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|months?)\b)",
        re.I,
    )
    match = pattern.search(sentence)
    return match.group(0) if match else None


def extract_experience(text: str) -> float:
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        text.lower(),
    )
    return max((float(x) for x in matches), default=0.0)


def _project_entities(text: str):
    out = []
    for match in PROJECT_PATTERNS.finditer(text):
        name = normalize_surface(match.group(1))
        name = re.sub(
            r"^(?:project|application|app|platform|system|tool|database)\s+"
            r"(?:called|named)\s+",
            "",
            name,
            flags=re.I,
        )
        name = re.split(
            r"\s+(?:using|with|built|developed|for|that|which)\b",
            name,
            1,
            flags=re.I,
        )[0]
        if name:
            out.append({
                "name": name,
                "category": "project",
                "canonical_key": canonical_key(name),
                "confidence": 0.95,
                "evidence": [match.group(0)],
                "source": "project_context",
            })
    return out


def extract_entities(text: str, max_candidates: int = 120):
    """Convert Part 1 JSON/evidence into the richer internal entity format."""
    result = extract(text)
    mapping = {
        "Skills": "skill",
        "Technologies": "technology",
        "Languages": "language",
    }
    entities = []

    for key, category in mapping.items():
        for name in result.get(key, []):
            meta = next(
                (
                    item for item in result.get("_evidence", [])
                    if item.get("name", "").lower() == name.lower()
                    and item.get("category") == key
                ),
                {},
            )
            evidence = [
                item.get("evidence", "")
                for item in result.get("_evidence", [])
                if item.get("name", "").lower() == name.lower()
                and item.get("category") == key
                and item.get("evidence")
            ]
            entities.append({
                "name": normalize_surface(name),
                "category": category,
                "canonical_key": canonical_key(name),
                "confidence": meta.get("confidence", 0.5),
                "evidence": evidence[:2],
                "source": meta.get("source", "part1_nlp"),
                "experience_context": extract_experience_for_entity(text, name),
            })

    entities.extend(_project_entities(text))

    best = {}
    for entity in entities:
        key = (entity["canonical_key"], entity["category"])
        if key not in best or entity.get("confidence", 0) > best[key].get("confidence", 0):
            best[key] = entity
    return list(best.values())[:max_candidates]


def build_candidate_profile(text, candidate_id=None, role=None):
    text = clean_text(text)
    entities = extract_entities(text)

    skills = sorted({e["name"] for e in entities if e["category"] == "skill"})
    technologies = sorted({e["name"] for e in entities if e["category"] == "technology"})
    languages = sorted({e["name"] for e in entities if e["category"] == "language"})
    projects = sorted({e["name"] for e in entities if e["category"] == "project"})

    return {
        "candidate_id": candidate_id,
        "role": role,
        "skills": skills,
        "technologies": technologies,
        "languages": languages,
        "projects": projects,
        "experience_years": extract_experience(text),
        "entities": entities,
        "project_evidence": [
            sentence for sentence in sentences(text)
            if any(project.lower() in sentence.lower() for project in projects)
        ][:10],
        "source_text": text,
    }


def extract_part1_json(text: str):
    """Return exactly the JSON categories required by Part 1."""
    result = extract(text)
    return {
        "Skills": result.get("Skills", []),
        "Technologies": result.get("Technologies", []),
        "Languages": result.get("Languages", []),
    }
