"""
preprocessing.py
----------------
Robust text preprocessing pipeline for NLP classification.
Handles tokenization, stop-word removal, lemmatization, and cleaning.
Works without NLTK downloads by bundling its own stop-word list.
"""

import re
import string
from typing import List

# ── Built-in English stop-word list (no NLTK download required) ──────────────
ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn", "rt", "via", "amp", "get", "got", "go", "going",
    "u", "ur", "r", "like", "im", "would", "could", "said", "one", "also",
}

# Try to extend with NLTK if available
try:
    from nltk.corpus import stopwords as nltk_sw
    ENGLISH_STOPWORDS |= set(nltk_sw.words("english"))
except Exception:
    pass


def _simple_lemmatize(word: str) -> str:
    """
    Rule-based lemmatizer (no NLTK WordNetLemmatizer required).
    Handles common English suffixes.
    """
    if len(word) <= 3:
        return word
    suffixes = [
        ("ness", ""), ("ment", ""), ("tion", ""), ("sion", ""),
        ("ing", ""), ("ings", ""), ("ed", ""), ("er", ""), ("ers", ""),
        ("ies", "y"), ("ied", "y"), ("ly", ""), ("ful", ""), ("less", ""),
        ("able", ""), ("ible", ""), ("al", ""), ("ous", ""), ("ive", ""),
        ("ize", ""), ("ise", ""), ("isation", ""), ("ization", ""),
    ]
    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    # Plural: remove trailing 's' carefully
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def clean_text(text: str) -> str:
    """
    Stage 1 – Raw text cleaning:
      • Lowercase
      • Remove URLs, mentions, hashtag symbols, HTML tags
      • Remove numbers (keep only alphabetic tokens)
      • Collapse whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                     # @mentions
    text = re.sub(r"#(\w+)", r" \1", text)                # #hashtag → word
    text = re.sub(r"<[^>]+>", " ", text)                  # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer (no NLTK punkt required)."""
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in ENGLISH_STOPWORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [_simple_lemmatize(t) for t in tokens]


def preprocess(text: str, min_len: int = 2) -> str:
    """
    Full pipeline:
        raw text → clean → tokenize → remove stop-words → lemmatize → rejoin
    Returns a single preprocessed string ready for vectorisation.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = [t for t in tokens if len(t) >= min_len]
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = [t for t in tokens if len(t) >= min_len]  # post-lemmatize filter
    return " ".join(tokens)


def preprocess_batch(texts: List[str], verbose: bool = False) -> List[str]:
    """Vectorised preprocessing over a list of texts."""
    results = []
    n = len(texts)
    for i, t in enumerate(texts):
        results.append(preprocess(t))
        if verbose and (i + 1) % 10_000 == 0:
            print(f"  Preprocessed {i+1}/{n} texts…")
    return results
