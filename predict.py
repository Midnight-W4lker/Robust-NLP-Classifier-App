"""
predict.py
----------
Loads trained models and provides a unified prediction interface
for TF-IDF, Word2Vec, and GloVe-style classifiers.
Word2Vec/GloVe models are optional - app works with TF-IDF if gensim unavailable.
"""

import os, joblib
import numpy as np
import pandas as pd

# Try to import gensim, but make it optional
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️ Warning: gensim not available. Word2Vec and GloVe models disabled.")

from preprocessing import preprocess_batch
from column_detector import detect_columns

BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _doc_vector(tokens, model, size):
    if not GENSIM_AVAILABLE:
        return np.zeros(size)
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(size)


def _sentences_to_matrix(sentences, model, size):
    if not GENSIM_AVAILABLE:
        return np.zeros((len(sentences), size))
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
        # TF-IDF is always required
        tfidf_needed = ["tfidf_vectorizer.pkl", "tfidf_clf.pkl"]
        tfidf_exist = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in tfidf_needed)
        
        # Word2Vec/GloVe only checked if gensim is available
        if GENSIM_AVAILABLE:
            w2v_needed = ["word2vec.model", "word2vec_clf.pkl", "glove_style.model", "glove_clf.pkl"]
            return tfidf_exist and all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in w2v_needed)
        else:
            # Only TF-IDF required if gensim unavailable
            return tfidf_exist

    @classmethod
    def load_all(cls):
        if not cls._models_exist():
            raise FileNotFoundError(
                "Trained models not found. Please run train_models.py first."
            )
        if cls._tfidf_vec is None:
            # Always load TF-IDF
            cls._tfidf_vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
            cls._tfidf_clf = joblib.load(os.path.join(MODELS_DIR, "tfidf_clf.pkl"))
            
            # Only load Word2Vec/GloVe if gensim is available
            if GENSIM_AVAILABLE:
                try:
                    cls._w2v_model = Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
                    cls._w2v_clf   = joblib.load(os.path.join(MODELS_DIR, "word2vec_clf.pkl"))
                    cls._glv_model = Word2Vec.load(os.path.join(MODELS_DIR, "glove_style.model"))
                    cls._glv_clf   = joblib.load(os.path.join(MODELS_DIR, "glove_clf.pkl"))
                except Exception as e:
                    print(f"⚠️ Warning: Could not load Word2Vec/GloVe models: {e}")

    @classmethod
    def models_ready(cls) -> bool:
        return cls._models_exist()
    
    @classmethod
    def is_model_available(cls, model_name: str) -> bool:
        """Check if a specific model is available."""
        if model_name == "tfidf":
            return os.path.exists(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        elif model_name in ["word2vec", "glove"]:
            if not GENSIM_AVAILABLE:
                return False
            model_file = "word2vec.model" if model_name == "word2vec" else "glove_style.model"
            return os.path.exists(os.path.join(MODELS_DIR, model_file))
        return False


def predict_texts(texts: list, model_name: str) -> dict:
    """
    Run prediction on a list of raw texts.
    model_name: one of 'tfidf' | 'word2vec' | 'glove'
    Returns dict with keys: labels, probabilities, correct, incorrect (if ground truth provided)
    """
    # Check if model is available
    if model_name in ["word2vec", "glove"] and not GENSIM_AVAILABLE:
        raise ValueError(f"{model_name} requires gensim, which is not available. Please use 'tfidf' model.")
    
    ModelLoader.load_all()
    processed = preprocess_batch(texts)

    if model_name == "tfidf":
        X    = ModelLoader._tfidf_vec.transform(processed)
        clf  = ModelLoader._tfidf_clf

    elif model_name == "word2vec":
        if not GENSIM_AVAILABLE or ModelLoader._w2v_model is None:
            raise ValueError("Word2Vec model not available. Please use 'tfidf' model.")
        tokens = [t.split() for t in processed]
        size   = ModelLoader._w2v_model.vector_size
        X      = _sentences_to_matrix(tokens, ModelLoader._w2v_model, size)
        clf    = ModelLoader._w2v_clf

    elif model_name == "glove":
        if not GENSIM_AVAILABLE or ModelLoader._glv_model is None:
            raise ValueError("GloVe model not available. Please use 'tfidf' model.")
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


def predict_file(df: pd.DataFrame, model_name: str) -> dict:
    """
    Run predictions on a DataFrame (supports both supervised and unsupervised data).
    For unsupervised data (no labels), only returns predictions.
    For supervised data (with labels), returns predictions + evaluation.
    """
    # Auto-detect columns
    detected = detect_columns(df)
    
    text_col = detected['text']
    if not text_col:
        raise ValueError(f"Could not auto-detect text column. Found columns: {list(df.columns)}")
    
    texts = df[text_col].astype(str).tolist()
    result = predict_texts(texts, model_name)
    preds = result["predictions"]
    probs = result["probabilities"]
    
    # Build base response
    response = {
        "model": model_name,
        "total": len(texts),
        "is_supervised": detected['is_supervised'],
        "text_column": text_col,
        "label_column": detected['label'],
    }
    
    # If supervised, add evaluation metrics
    if detected['is_supervised']:
        label_col = detected['label']
        
        # Map string labels to integers if needed
        if df[label_col].dtype == object:
            label_map = {"negative": 0, "neutral": 1, "positive": 1}
            unique_labels = set(df[label_col].dropna().unique())
            if unique_labels <= {"negative", "positive"}:
                label_map = {"negative": 0, "positive": 1}
            elif unique_labels <= {"negative", "neutral", "positive"}:
                label_map = {"negative": 0, "neutral": 1, "positive": 2}
            df[label_col] = df[label_col].map(label_map)
        labels = df[label_col].astype(int).tolist()
        
        correct = sum(1 for p, g in zip(preds, labels) if p == g)
        incorrect = sum(1 for p, g in zip(preds, labels) if p != g)
        accuracy = correct / len(labels) if labels else 0
        
        response.update({
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": round(accuracy, 4),
        })
        
        # Build per-row detail with labels
        rows = []
        for i, (text, label, pred, prob) in enumerate(zip(texts, labels, preds, probs)):
            rows.append({
                "id": i + 1,
                "text": text[:120] + ("…" if len(text) > 120 else ""),
                "true_label": label,
                "predicted": pred,
                "correct": pred == label,
                "confidence": round(float(prob), 4) if prob is not None else None,
            })
    else:
        # Unsupervised: no labels, just predictions
        rows = []
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            rows.append({
                "id": i + 1,
                "text": text[:120] + ("…" if len(text) > 120 else ""),
                "predicted": pred,
                "confidence": round(float(prob), 4) if prob is not None else None,
            })
    
    response["details"] = rows
    return response


def evaluate_file(df: pd.DataFrame, model_name: str) -> dict:
    """
    Evaluate predictions on a DataFrame with auto-detected text and label columns.
    Returns full results including correct/incorrect breakdown.
    """
    # Auto-detect text and label columns using intelligent detection
    detected = detect_columns(df)
    
    text_col = detected['text']
    label_col = detected['label']
    
    if not text_col:
        raise ValueError(f"Could not auto-detect text column. Found columns: {list(df.columns)}")
    
    if not label_col:
        raise ValueError(f"Could not auto-detect label column. This appears to be unsupervised data. Found columns: {list(df.columns)}")
    
    texts  = df[text_col].astype(str).tolist()
    
    # Map string labels to integers if needed
    if df[label_col].dtype == object:
        label_map = {"negative": 0, "neutral": 1, "positive": 1}
        unique_labels = set(df[label_col].dropna().unique())
        if unique_labels <= {"negative", "positive"}:
            label_map = {"negative": 0, "positive": 1}
        elif unique_labels <= {"negative", "neutral", "positive"}:
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df[label_col] = df[label_col].map(label_map)
    labels = df[label_col].astype(int).tolist()

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
