"""
api.py
------
Flask REST API exposing the three classifiers.

Endpoints:
  GET  /health                  → server status & model readiness
  POST /predict/single          → classify one text string
  POST /predict/batch           → classify a list of texts
  POST /evaluate                → upload CSV, get correct/incorrect breakdown
  GET  /models/status           → which models are loaded
  GET  /results/summary         → training results from models/training_results.json
"""

import os, json, io
import pandas as pd
from flask import Flask, request, jsonify

from predict import predict_single, evaluate_file, ModelLoader

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_ready": ModelLoader.models_ready(),
    })


# ── Single prediction ─────────────────────────────────────────────────────────
@app.route("/predict/single", methods=["POST"])
def predict_single_endpoint():
    """
    Body: { "text": "...", "model": "tfidf" | "word2vec" | "glove" }
    """
    data  = request.get_json(force=True)
    text  = data.get("text", "")
    model = data.get("model", "tfidf")

    if not text.strip():
        return jsonify({"error": "text cannot be empty"}), 400
    if model not in ("tfidf", "word2vec", "glove"):
        return jsonify({"error": "model must be tfidf, word2vec, or glove"}), 400

    try:
        result = predict_single(text, model)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Batch prediction ──────────────────────────────────────────────────────────
@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    """
    Body: { "texts": ["...", "..."], "model": "tfidf" }
    """
    from predict import predict_texts
    data   = request.get_json(force=True)
    texts  = data.get("texts", [])
    model  = data.get("model", "tfidf")

    if not texts:
        return jsonify({"error": "texts list cannot be empty"}), 400

    try:
        result = predict_texts(texts, model)
        return jsonify({
            "model": model,
            "count": len(texts),
            "predictions": result["predictions"],
            "probabilities": result["probabilities"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Evaluate CSV ───────────────────────────────────────────────────────────────
@app.route("/evaluate", methods=["POST"])
def evaluate_endpoint():
    """
    Multipart form:
      file  – CSV with 'text' and 'label' columns
      model – tfidf | word2vec | glove
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model = request.form.get("model", "tfidf")
    f     = request.files["file"]
    try:
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8", errors="replace")))
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    if "text" not in df.columns or "label" not in df.columns:
        return jsonify({"error": "CSV must have 'text' and 'label' columns"}), 400

    df = df.dropna(subset=["text", "label"])

    try:
        result = evaluate_file(df, model)
        # Strip full details for large files to keep response size reasonable
        if len(result["details"]) > 200:
            result["details_sample"] = result["details"][:200]
            del result["details"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Model status ───────────────────────────────────────────────────────────────
@app.route("/models/status", methods=["GET"])
def models_status():
    return jsonify({
        "ready": ModelLoader.models_ready(),
        "models": ["tfidf", "word2vec", "glove"],
    })


# ── Training summary ───────────────────────────────────────────────────────────
@app.route("/results/summary", methods=["GET"])
def results_summary():
    path = os.path.join(BASE_DIR, "models", "training_results.json")
    if not os.path.exists(path):
        return jsonify({"error": "Training results not found"}), 404
    with open(path) as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    print("Starting NLP Classifier API on http://0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
