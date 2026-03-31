"""
predict.py
----------
Loads trained models and provides a unified prediction interface
for TF-IDF, Word2Vec, and GloVe-style classifiers.
"""

import os, joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from preprocessing import preprocess_batch

BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _doc_vector(tokens, model, size):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(size)


def _sentences_to_matrix(sentences, model, size):
    return np.array([_doc_vector(s, model, size) for s in sentences])


class ModelLoader:
    """Lazy-loads models once and caches them."""
    _tfidf_vec = None
    _tfidf_clf = None
    _w2v_model = None
    _w2v_clf   = None
    _glv_model = None
    _glv_clf   = None

    @classmethod
    def _models_exist(cls) -> bool:
        needed = [
            "tfidf_vectorizer.pkl", "tfidf_clf.pkl",
            "word2vec.model", "word2vec_clf.pkl",
            "glove_style.model", "glove_clf.pkl",
        ]
        return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in needed)

    @classmethod
    def load_all(cls):
        if not cls._models_exist():
            raise FileNotFoundError(
                "Trained models not found. Please run train_models.py first."
            )
        if cls._tfidf_vec is None:
            cls._tfidf_vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
            cls._tfidf_clf = joblib.load(os.path.join(MODELS_DIR, "tfidf_clf.pkl"))
            cls._w2v_model = Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
            cls._w2v_clf   = joblib.load(os.path.join(MODELS_DIR, "word2vec_clf.pkl"))
            cls._glv_model = Word2Vec.load(os.path.join(MODELS_DIR, "glove_style.model"))
            cls._glv_clf   = joblib.load(os.path.join(MODELS_DIR, "glove_clf.pkl"))

    @classmethod
    def models_ready(cls) -> bool:
        return cls._models_exist()


def predict_texts(texts: list, model_name: str) -> dict:
    """
    Run prediction on a list of raw texts.
    model_name: one of 'tfidf' | 'word2vec' | 'glove'
    Returns dict with keys: labels, probabilities, correct, incorrect (if ground truth provided)
    """
    ModelLoader.load_all()
    processed = preprocess_batch(texts)

    if model_name == "tfidf":
        X    = ModelLoader._tfidf_vec.transform(processed)
        clf  = ModelLoader._tfidf_clf

    elif model_name == "word2vec":
        tokens = [t.split() for t in processed]
        size   = ModelLoader._w2v_model.vector_size
        X      = _sentences_to_matrix(tokens, ModelLoader._w2v_model, size)
        clf    = ModelLoader._w2v_clf

    elif model_name == "glove":
        tokens = [t.split() for t in processed]
        size   = ModelLoader._glv_model.vector_size
        X      = _sentences_to_matrix(tokens, ModelLoader._glv_model, size)
        clf    = ModelLoader._glv_clf

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose tfidf, word2vec, or glove.")

    preds = clf.predict(X)
    try:
        probs = clf.predict_proba(X).max(axis=1).tolist()
    except Exception:
        probs = [None] * len(preds)

    return {
        "predictions": preds.tolist(),
        "probabilities": probs,
        "processed_texts": processed,
    }


def evaluate_file(df: pd.DataFrame, model_name: str) -> dict:
    """
    Evaluate predictions on a DataFrame with 'text' and 'label' columns.
    Returns full results including correct/incorrect breakdown.
    """
    texts  = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    result  = predict_texts(texts, model_name)
    preds   = result["predictions"]
    probs   = result["probabilities"]

    correct   = sum(1 for p, g in zip(preds, labels) if p == g)
    incorrect = sum(1 for p, g in zip(preds, labels) if p != g)
    accuracy  = correct / len(labels) if labels else 0

    # Build per-row detail
    rows = []
    for i, (text, label, pred, prob) in enumerate(
        zip(texts, labels, preds, probs)
    ):
        rows.append({
            "id": i + 1,
            "text": text[:120] + ("…" if len(text) > 120 else ""),
            "true_label": label,
            "predicted": pred,
            "correct": pred == label,
            "confidence": round(float(prob), 4) if prob is not None else None,
        })

    return {
        "model": model_name,
        "total": len(labels),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(accuracy, 4),
        "details": rows,
    }


def predict_single(text: str, model_name: str) -> dict:
    """Predict a single text snippet."""
    result = predict_texts([text], model_name)
    return {
        "prediction": result["predictions"][0],
        "confidence": round(float(result["probabilities"][0]), 4)
            if result["probabilities"][0] is not None else None,
        "processed": result["processed_texts"][0],
        "model": model_name,
        "label_name": "Positive" if result["predictions"][0] == 1 else "Negative",
    }
