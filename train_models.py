"""
train_models.py
---------------
Trains three classifiers on the primary dataset:
  1. TF-IDF  + Logistic Regression
  2. Word2Vec + Logistic Regression (averaged embeddings)
  3. GloVe-style (trained in-house via gensim) + Logistic Regression

All trained artefacts are saved to models/ so the API and Streamlit app
can load them without re-training.
"""

import os, json, time, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix)
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

from preprocessing import preprocess_batch

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_train_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "train_primary.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df):,} training rows. Label dist:\n{df['label'].value_counts().to_dict()}")
    return df


def doc_vector(tokens: list, model, vector_size: int) -> np.ndarray:
    """Average Word2Vec / GloVe vectors for a tokenised document."""
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(vector_size)


def sentences_to_matrix(sentences: list, model, vector_size: int) -> np.ndarray:
    return np.array([doc_vector(s, model, vector_size) for s in sentences])


def evaluate(model_name, clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    correct   = int((preds == y_test).sum())
    incorrect = int((preds != y_test).sum())
    print(f"\n── {model_name} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Correct  : {correct}  |  Incorrect: {incorrect}")
    print(classification_report(y_test, preds))
    return {
        "accuracy": acc,
        "correct": correct,
        "incorrect": incorrect,
        "report": report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. TF-IDF model
# ─────────────────────────────────────────────────────────────────────────────
def train_tfidf(X_train_text, y_train, X_val_text, y_val):
    print("\n[1/3] Training TF-IDF + Logistic Regression…")
    t0 = time.time()

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_val   = vectorizer.transform(X_val_text)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="saga", n_jobs=-1)
    clf.fit(X_train, y_train)

    metrics = evaluate("TF-IDF", clf, X_val, y_val)
    metrics["train_time"] = round(time.time() - t0, 2)

    # Save artefacts
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(clf,        os.path.join(MODELS_DIR, "tfidf_clf.pkl"))
    print(f"  Saved TF-IDF model. ({metrics['train_time']}s)")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Word2Vec model
# ─────────────────────────────────────────────────────────────────────────────
def train_word2vec(X_train_text, y_train, X_val_text, y_val):
    print("\n[2/3] Training Word2Vec + Logistic Regression…")
    t0 = time.time()

    VECTOR_SIZE = 100
    train_tokens = [t.split() for t in X_train_text]
    val_tokens   = [t.split() for t in X_val_text]

    w2v = Word2Vec(
        sentences=train_tokens,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=4,
        epochs=5,
        sg=0,            # CBOW
    )
    w2v.save(os.path.join(MODELS_DIR, "word2vec.model"))

    X_train = sentences_to_matrix(train_tokens, w2v, VECTOR_SIZE)
    X_val   = sentences_to_matrix(val_tokens,   w2v, VECTOR_SIZE)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="saga", n_jobs=-1)
    clf.fit(X_train, y_train)

    metrics = evaluate("Word2Vec", clf, X_val, y_val)
    metrics["train_time"]   = round(time.time() - t0, 2)
    metrics["vector_size"]  = VECTOR_SIZE

    joblib.dump(clf, os.path.join(MODELS_DIR, "word2vec_clf.pkl"))
    print(f"  Saved Word2Vec model. ({metrics['train_time']}s)")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3. GloVe-style model (trained with gensim Word2Vec using skip-gram)
#    We call it "GloVe-style" because true GloVe requires a separate C
#    binary. Here we use a skip-gram Word2Vec trained on the same corpus,
#    which produces comparable dense embeddings.
# ─────────────────────────────────────────────────────────────────────────────
def train_glove_style(X_train_text, y_train, X_val_text, y_val):
    print("\n[3/3] Training GloVe-style (Skip-gram) + Logistic Regression…")
    t0 = time.time()

    VECTOR_SIZE = 150
    train_tokens = [t.split() for t in X_train_text]
    val_tokens   = [t.split() for t in X_val_text]

    glove_model = Word2Vec(
        sentences=train_tokens,
        vector_size=VECTOR_SIZE,
        window=10,
        min_count=2,
        workers=4,
        epochs=8,
        sg=1,            # Skip-gram (closer to GloVe behaviour)
        hs=0,
        negative=5,
    )
    glove_model.save(os.path.join(MODELS_DIR, "glove_style.model"))

    X_train = sentences_to_matrix(train_tokens, glove_model, VECTOR_SIZE)
    X_val   = sentences_to_matrix(val_tokens,   glove_model, VECTOR_SIZE)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="saga", n_jobs=-1)
    clf.fit(X_train, y_train)

    metrics = evaluate("GloVe-style", clf, X_val, y_val)
    metrics["train_time"]  = round(time.time() - t0, 2)
    metrics["vector_size"] = VECTOR_SIZE

    joblib.dump(clf, os.path.join(MODELS_DIR, "glove_clf.pkl"))
    print(f"  Saved GloVe-style model. ({metrics['train_time']}s)")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def train_all():
    df = load_train_data()

    print("\nPreprocessing texts…")
    df["processed"] = preprocess_batch(df["text"].tolist(), verbose=True)

    X = df["processed"].tolist()
    y = df["label"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    results = {}
    results["tfidf"]       = train_tfidf(X_train, y_train, X_val, y_val)
    results["word2vec"]    = train_word2vec(X_train, y_train, X_val, y_val)
    results["glove_style"] = train_glove_style(X_train, y_train, X_val, y_val)

    # Save summary
    summary_path = os.path.join(MODELS_DIR, "training_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ All models trained. Summary saved → {summary_path}")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              TRAINING SUMMARY (Validation Set)          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for name, m in results.items():
        print(f"║  {name:<12}  Accuracy: {m['accuracy']:.4f}  "
              f"Correct: {m['correct']:>6}  Incorrect: {m['incorrect']:>6}  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return results


if __name__ == "__main__":
    train_all()
