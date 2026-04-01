"""
streamlit_app.py
----------------
Streamlit interface for the NLP Text Classifier.
Allows examiners to:
  • Upload a test CSV and see real-time classification results
  • Compare TF-IDF, Word2Vec, and GloVe-style models side-by-side
  • View Correct vs Incorrect predictions with confidence scores
  • Try a live single-text prediction demo
  • View training summary and comparison table
"""

import os, sys, json, io, time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Import column detection utility ──────────────────────────────────────────
from column_detector import detect_columns, validate_dataset, get_column_info

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Text Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    .sub-header { 
        color: #6c757d; 
        font-size: 1rem; 
        margin-top: 0; 
    }
    .metric-box {
        background: #f8f9fa; 
        border-radius: 12px; 
        padding: 16px 20px;
        border-left: 5px solid #667eea; 
        margin: 8px 0;
        color: #000;
    }
    .correct-badge { 
        background:#d4edda; 
        color:#155724; 
        border-radius:6px; 
        padding:2px 8px; 
        font-weight:600; 
    }
    .incorrect-badge { 
        background:#f8d7da; 
        color:#721c24; 
        border-radius:6px; 
        padding:2px 8px; 
        font-weight:600; 
    }
    .model-card {
        background: rgba(255, 255, 255, 0.05); 
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; 
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); 
        margin: 8px 0;
        color: inherit;
    }
    .model-card h3, .model-card h4 {
        color: inherit;
        margin-top: 0;
    }
    .model-card p, .model-card strong, .model-card em {
        color: inherit;
        opacity: 0.9;
    }
    .stDataFrame { 
        border-radius: 10px; 
    }
    div[data-testid="stTabs"] button { 
        font-weight: 600; 
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    from predict import ModelLoader
    ModelLoader.load_all()
    return True


def check_models_ready():
    from predict import ModelLoader
    return ModelLoader.models_ready()


def get_available_models():
    """Get list of available models based on what's installed."""
    from predict import ModelLoader, GENSIM_AVAILABLE
    available = []
    
    if ModelLoader.is_model_available("tfidf"):
        available.append("tfidf")
    
    if GENSIM_AVAILABLE:
        if ModelLoader.is_model_available("word2vec"):
            available.append("word2vec")
        if ModelLoader.is_model_available("glove"):
            available.append("glove")
    
    return available if available else ["tfidf"]  # Default to tfidf


def check_gensim_available():
    """Check if gensim is available."""
    from predict import GENSIM_AVAILABLE
    return GENSIM_AVAILABLE


def run_evaluation(df: pd.DataFrame, model_name: str) -> dict:
    from predict import evaluate_file
    return evaluate_file(df, model_name)


def run_single(text: str, model_name: str) -> dict:
    from predict import predict_single
    return predict_single(text, model_name)


MODEL_LABELS = {
    "tfidf":    "TF-IDF",
    "word2vec": "Word2Vec",
    "glove":    "GloVe-style",
}
MODEL_COLORS = {
    "tfidf":    "#667eea",
    "word2vec": "#f59e0b",
    "glove":    "#10b981",
}


def accuracy_gauge(accuracy: float, model_name: str, color: str):
    fig, ax = plt.subplots(figsize=(2.8, 1.8))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    bar = ax.barh([0], [accuracy * 100], color=color, height=0.5, zorder=3)
    ax.barh([0], [100], color="#e9ecef", height=0.5, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.text(accuracy * 100 + 1, 0, f"{accuracy*100:.1f}%",
            va="center", fontsize=11, fontweight="bold", color=color)
    ax.set_title(model_name, fontsize=10, pad=4)
    return fig


def bar_chart_correct_incorrect(results: dict):
    """Grouped bar chart for all three models."""
    models = list(results.keys())
    correct   = [results[m]["correct"]   for m in models]
    incorrect = [results[m]["incorrect"] for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("#fafafa")
    b1 = ax.bar(x - w/2, correct,   w, label="Correct",   color="#10b981", zorder=3)
    b2 = ax.bar(x + w/2, incorrect, w, label="Incorrect", color="#ef4444", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=12)
    ax.set_ylabel("Number of Predictions")
    ax.set_title("Correct vs Incorrect Predictions by Model", fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


def accuracy_comparison_bar(results: dict):
    models = list(results.keys())
    accs   = [results[m]["accuracy"] * 100 for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("#fafafa")
    bars = ax.barh([MODEL_LABELS[m] for m in models], accs, color=colors, zorder=3)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NLP Classifier")
    st.markdown("---")

    ready = check_models_ready()
    if ready:
        st.success("✅ Models Ready")
        try:
            load_models()
        except Exception as e:
            st.error(f"Load error: {e}")
    else:
        st.warning("⚠️ Models not trained yet")
        st.info("Run `python train_models.py` to train the models.")
    
    # Check gensim availability
    if not check_gensim_available():
        st.warning("⚠️ Word2Vec/GloVe unavailable (gensim missing)")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📁 Test File Upload", "🔍 Live Demo", "📊 Model Comparison", "📋 Training Summary"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Available Models:**")
    available_models = get_available_models()
    if "tfidf" in available_models:
        st.markdown("- 🔵 TF-IDF")
    if "word2vec" in available_models:
        st.markdown("- 🟡 Word2Vec (CBOW)")
    if "glove" in available_models:
        st.markdown("- 🟢 GloVe-style (Skip-gram)")
    st.markdown("---")
    st.caption("Lab Mid: Robust Text Classification\nNLP Course — 2024/25")


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-header">🧠 Robust NLP Text Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Multi-Model Text Classification with Auto-Detection</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    
    # ── Key Features Overview ──
    st.markdown("### ✨ Key Features")
    feat1, feat2, feat3 = st.columns(3)
    with feat1:
        st.markdown("""
        <div class='model-card' style='border-left: 5px solid #667eea;'>
        <h4>🎯 Smart Detection</h4>
        <p>Automatically detects text and label columns from any CSV format. 
        Works with diverse naming conventions and supports both supervised and unsupervised datasets.</p>
        </div>
        """, unsafe_allow_html=True)
    with feat2:
        st.markdown("""
        <div class='model-card' style='border-left: 5px solid #f59e0b;'>
        <h4>🚀 Multiple Models</h4>
        <p>Compare TF-IDF, Word2Vec, and GloVe-style embeddings side-by-side.
        Get instant accuracy metrics and detailed prediction breakdowns.</p>
        </div>
        """, unsafe_allow_html=True)
    with feat3:
        st.markdown("""
        <div class='model-card' style='border-left: 5px solid #10b981;'>
        <h4>📊 Real-time Results</h4>
        <p>Upload your data and get predictions in seconds.
        Interactive visualizations show correct vs incorrect predictions with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # ── Model Architectures ──
    st.markdown("### 🤖 Model Architectures")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='model-card'>
        <h3>🔵 TF-IDF</h3>
        <p><strong>Term Frequency–Inverse Document Frequency</strong><br><br>
        • Sparse bag-of-words representation<br>
        • N-gram support (unigrams + bigrams)<br>
        • 50,000 feature vocabulary<br>
        • Logistic Regression classifier<br>
        <br><em>Fast, interpretable baseline</em></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='model-card'>
        <h3>🟡 Word2Vec (CBOW)</h3>
        <p><strong>Continuous Bag-of-Words</strong><br><br>
        • Neural word embeddings<br>
        • Trained on corpus<br>
        • Average token vectors<br>
        • Captures semantic similarity<br>
        <br><em>Context-aware semantics</em></p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='model-card'>
        <h3>🟢 GloVe-style</h3>
        <p><strong>Skip-gram with Large Context</strong><br><br>
        • Skip-gram Word2Vec variant<br>
        • Larger context window (10)<br>
        • Better rare word handling<br>
        • Long-range dependencies<br>
        <br><em>Enhanced word relationships</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # ── Intelligent Column Detection ──
    st.markdown("### 🎯 Intelligent Column Detection")
    st.info("**No need to rename your columns!** The app automatically detects text and label columns from your CSV.")
    
    det1, det2 = st.columns(2)
    with det1:
        st.markdown("""
        **📝 Text Column Detection:**
        - Recognizes: `text`, `review`, `content`, `sentence`, `message`, `tweet`, `comment`, `description`, etc.
        - Analyzes text length (prefers >20 chars avg)
        - Identifies string columns with longest content
        """)
    with det2:
        st.markdown("""
        **🏷️ Label Column Detection** *(optional)*:
        - Recognizes: `label`, `sentiment`, `class`, `category`, `target`, `rating`, etc.
        - Analyzes cardinality (2-50 unique values)
        - Supports unsupervised data (no labels)
        """)

    st.markdown("---")
    
    # ── Dataset Support ──
    st.markdown("### 📂 Supported Datasets")
    data_col1, data_col2 = st.columns(2)
    with data_col1:
        st.markdown("""
        **✅ Supervised Data** (with labels):
        - Get full accuracy metrics
        - Correct vs incorrect breakdown
        - Confidence scores per prediction
        - Filter by correct/incorrect results
        """)
    with data_col2:
        st.markdown("""
        **🔍 Unsupervised Data** (text only):
        - Get predictions without labels
        - Confidence scores
        - Class distribution
        - Export predictions to CSV
        """)

    st.markdown("---")
    st.markdown("### ⚙️ Preprocessing Pipeline")
    st.caption("Robust text cleaning without external dependencies")
    steps = [
        ("1. Lowercasing", "Normalizes case variation"),
        ("2. URL/mention removal", "Removes @user, http:// noise"),
        ("3. Hashtag expansion", "#AI → AI"),
        ("4. HTML tag stripping", "Cleans web-scraped text"),
        ("5. Non-alpha removal", "Keeps only letters"),
        ("6. Tokenization", "Whitespace split (no NLTK)"),
        ("7. Stop-word removal", "500+ English stop words"),
        ("8. Rule-based lemmatization", "Strips common suffixes"),
        ("9. Min-length filter", "Removes tokens < 2 chars"),
    ]
    cols = st.columns(3)
    for i, (step, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"**{step}**  \n_{desc}_")
    
    st.markdown("---")
    
    # ── Quick Start Guide ──
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **📁 Upload Dataset**: Go to "Test File Upload" and upload your CSV
    2. **🎯 Auto-Detection**: App detects text and label columns automatically
    3. **🤖 Select Models**: Choose which models to compare (or use all three)
    4. **▶️ Run**: Click "Run Classification" and get instant results
    5. **📊 Analyze**: View accuracy, predictions, and download results
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Training Data")
    st.caption("Models pre-trained on diverse datasets for robust generalization")
    info_df = pd.DataFrame({
        "Dataset": ["Tweets (Sentiment140-style)", "Movie Reviews (IMDb-style)",
                    "Titanic Text", "Trade & News Headlines"],
        "Size": ["80,000 train / 5,000 test", "30,000 train / 3,000 test",
                 "5,000 train", "20,000 train / 3,000 test"],
        "Type": ["Binary Sentiment", "Binary Sentiment", "Binary Survival", "5-class Topic"],
        "Use": ["Primary Train + Test 1", "Primary Train + Test 2", "Primary Train", "Test 2"],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.caption("💡 **Tip**: Start with 'Test File Upload' to try the classifier on your own data!")



# ─────────────────────────────────────────────────────────────────────────────
# TEST FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Test File Upload":
    st.markdown("## 📁 Upload Test Dataset")
    st.info("Upload a CSV file. The app will auto-detect text and label columns (if available).\n\n✅ **Supervised data** (with labels): Get accuracy and evaluation metrics\n\n🔍 **Unsupervised data** (no labels): Get predictions only")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    # Get available models
    available_models = get_available_models()
    has_gensim = check_gensim_available()
    
    if not has_gensim:
        st.warning("⚠️ **Note**: Word2Vec and GloVe models require gensim library which is not available. Only TF-IDF model is active.")
    
    models_sel = st.multiselect(
        "Select models to evaluate",
        options=available_models,
        default=available_models,
        format_func=lambda x: MODEL_LABELS[x],
    )

    if uploaded and models_sel:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**Preview** ({len(df_raw):,} rows)")
        st.dataframe(df_raw.head(5), use_container_width=True)

        # Auto-detect text and label columns using intelligent detection
        # For this page, we don't require labels (can be unsupervised)
        is_valid, msg, detected = validate_dataset(df_raw, require_labels=False)
        
        if not is_valid:
            st.error(msg)
            st.info("💡 **Tip**: Ensure your CSV has a text column (e.g., 'text', 'review', 'sentence', 'content')")
            
            # Show column analysis to help debug
            with st.expander("🔍 Column Analysis"):
                for col in df_raw.columns:
                    col_info = get_column_info(df_raw, col)
                    st.markdown(f"**{col_info['name']}**: {col_info['dtype']}, {col_info['n_unique']} unique values")
                    if 'avg_length' in col_info:
                        st.markdown(f"  - Avg length: {col_info['avg_length']} chars")
                    st.markdown(f"  - Sample: {col_info['sample_values'][:2]}")
            st.stop()
        
        st.success(msg)
        text_col = detected['text']
        label_col = detected['label']
        is_supervised = detected['is_supervised']
        
        # Clean data based on what we detected
        if is_supervised:
            df_raw = df_raw.dropna(subset=[text_col, label_col])
        else:
            df_raw = df_raw.dropna(subset=[text_col])
            st.warning("📌 **Unsupervised Mode**: No label column detected. Will show predictions only (no accuracy metrics).")

        if st.button("▶️ Run Classification", type="primary"):
            if not check_models_ready():
                st.error("Models not trained. Run train_models.py first.")
                st.stop()

            load_models()
            all_results = {}
            progress = st.progress(0, text="Running models…")

            # Import the new predict_file function that handles both supervised and unsupervised
            from predict import predict_file
            
            for i, mname in enumerate(models_sel):
                with st.spinner(f"Evaluating {MODEL_LABELS[mname]}…"):
                    t0 = time.time()
                    res = predict_file(df_raw.copy(), mname)
                    res["elapsed"] = round(time.time() - t0, 2)
                    all_results[mname] = res
                progress.progress((i + 1) / len(models_sel))

            progress.empty()
            st.success("✅ Classification complete!")

            # ── Summary metrics ──────────────────────────────────────────────
            if is_supervised:
                st.markdown("### 📊 Results Summary")
                cols = st.columns(len(models_sel))
                for col, mname in zip(cols, models_sel):
                    res = all_results[mname]
                    col.metric(
                        label=MODEL_LABELS[mname],
                        value=f"{res['accuracy']*100:.1f}%",
                        delta=f"{res['correct']} correct / {res['incorrect']} wrong",
                    )

                # ── Grouped bar chart ─────────────────────────────────────────────
                st.pyplot(bar_chart_correct_incorrect(all_results))
                st.pyplot(accuracy_comparison_bar(all_results))
            else:
                st.markdown("### 🔍 Prediction Summary")
                st.info("Unsupervised data - showing predictions only (no ground truth labels)")
                cols = st.columns(len(models_sel))
                for col, mname in zip(cols, models_sel):
                    res = all_results[mname]
                    # Show prediction distribution
                    pred_dist = pd.Series([d['predicted'] for d in res['details']]).value_counts()
                    col.metric(
                        label=MODEL_LABELS[mname],
                        value=f"{res['total']} predictions",
                        delta=f"Classes: {dict(pred_dist)}",
                    )

            # ── Comparison table ──────────────────────────────────────────────
            st.markdown("### 📋 Comparison Table")
            if is_supervised:
                cmp = pd.DataFrame({
                    "Model": [MODEL_LABELS[m] for m in models_sel],
                    "Total": [all_results[m]["total"]    for m in models_sel],
                    "Correct": [all_results[m]["correct"]   for m in models_sel],
                    "Incorrect": [all_results[m]["incorrect"] for m in models_sel],
                    "Accuracy (%)": [f"{all_results[m]['accuracy']*100:.2f}" for m in models_sel],
                    "Time (s)": [all_results[m]["elapsed"]   for m in models_sel],
                })
            else:
                cmp = pd.DataFrame({
                    "Model": [MODEL_LABELS[m] for m in models_sel],
                    "Total Predictions": [all_results[m]["total"] for m in models_sel],
                    "Time (s)": [all_results[m]["elapsed"] for m in models_sel],
                })
            st.dataframe(cmp, use_container_width=True, hide_index=True)

            # ── Per-row details ───────────────────────────────────────────────
            st.markdown("### 🔎 Per-Prediction Details")
            best_model = models_sel[0] if len(models_sel) == 1 else max(
                models_sel, key=lambda m: all_results[m].get("accuracy", 0)
            )
            details_df = pd.DataFrame(all_results[best_model]["details"])
            
            if is_supervised:
                details_df["result"] = details_df["correct"].map({True: "✅ Correct", False: "❌ Wrong"})
                st.caption(f"Showing results for best model: **{MODEL_LABELS[best_model]}**")

                filter_opt = st.radio("Filter", ["All", "✅ Correct only", "❌ Wrong only"], horizontal=True)
                if filter_opt == "✅ Correct only":
                    details_df = details_df[details_df["correct"]]
                elif filter_opt == "❌ Wrong only":
                    details_df = details_df[~details_df["correct"]]

                st.dataframe(
                    details_df[["id", "text", "true_label", "predicted", "result", "confidence"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption(f"Showing predictions from: **{MODEL_LABELS[best_model]}**")
                st.dataframe(
                    details_df[["id", "text", "predicted", "confidence"]],
                    use_container_width=True, hide_index=True,
                )

            # Download
            csv_out = cmp.to_csv(index=False)
            st.download_button("⬇️ Download Comparison Table", csv_out,
                               "comparison_results.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE DEMO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Live Demo":
    st.markdown("## 🔍 Live Single-Text Demo")
    st.markdown("Type or paste any text and see how each model classifies it in real-time.")

    sample_texts = [
        "This product is absolutely amazing and exceeded all my expectations!",
        "Terrible service, completely disappointed with the entire experience.",
        "The flight was delayed by three hours and the staff were unhelpful.",
        "Just had the best coffee of my life at this wonderful little café.",
        "Stock market rallies as inflation data shows signs of cooling.",
        "Clinical trial results show promising outcomes for new cancer treatment.",
    ]

    selected_sample = st.selectbox("Try a sample text:", ["(type your own)"] + sample_texts)
    user_text = st.text_area(
        "Enter text to classify:",
        value="" if selected_sample == "(type your own)" else selected_sample,
        height=120,
    )

    if st.button("🚀 Classify", type="primary") and user_text.strip():
        if not check_models_ready():
            st.error("Models not trained. Run train_models.py first.")
            st.stop()

        load_models()
        col1, col2, col3 = st.columns(3)
        for col, (mkey, mname) in zip(
            [col1, col2, col3],
            [("tfidf","TF-IDF"), ("word2vec","Word2Vec"), ("glove","GloVe-style")]
        ):
            with col:
                res = run_single(user_text, mkey)
                sentiment = res["label_name"]
                conf      = res["confidence"]
                color     = "#10b981" if sentiment == "Positive" else "#ef4444"
                emoji     = "😊" if sentiment == "Positive" else "😞"
                st.markdown(f"""
                <div class='model-card' style='border-left:5px solid {color}'>
                <h4>{mname}</h4>
                <p style='font-size:1.6rem;margin:0'>{emoji} <b style='color:{color}'>{sentiment}</b></p>
                <p>Confidence: <b>{conf*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("**Preprocessed text:**")
        from preprocessing import preprocess
        st.code(preprocess(user_text), language=None)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown("## 📊 Cross-Dataset Model Comparison")

    data_dir = os.path.join(BASE_DIR, "data")
    test_files = {
        "Test Dataset 1 (Tweets)": os.path.join(data_dir, "test_dataset1_tweets.csv"),
        "Test Dataset 2 (Mixed)": os.path.join(data_dir, "test_dataset2_mixed.csv"),
    }

    all_ok = all(os.path.exists(p) for p in test_files.values())
    if not all_ok:
        st.warning("Test datasets not found. Run `python generate_datasets.py` first.")
        st.stop()

    if not check_models_ready():
        st.error("Models not trained. Run `python train_models.py` first.")
        st.stop()

    if st.button("▶️ Run Full Cross-Dataset Evaluation", type="primary"):
        load_models()
        master = {}

        for ds_name, ds_path in test_files.items():
            df_test = pd.read_csv(ds_path).dropna(subset=["text","label"])
            master[ds_name] = {}
            for mname in ["tfidf", "word2vec", "glove"]:
                with st.spinner(f"{ds_name} – {MODEL_LABELS[mname]}…"):
                    res = run_evaluation(df_test.copy(), mname)
                    master[ds_name][mname] = res

        st.success("✅ Evaluation done!")

        # Full comparison table (the exam requirement)
        st.markdown("### 📋 Cross-Dataset Comparison Table")
        rows = []
        for ds_name in master:
            for mname in ["tfidf","word2vec","glove"]:
                r = master[ds_name][mname]
                rows.append({
                    "Dataset": ds_name,
                    "Model": MODEL_LABELS[mname],
                    "Total": r["total"],
                    "Correct ✅": r["correct"],
                    "Incorrect ❌": r["incorrect"],
                    "Accuracy (%)": f"{r['accuracy']*100:.2f}",
                })
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # Charts per dataset
        for ds_name in master:
            st.markdown(f"#### {ds_name}")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(bar_chart_correct_incorrect(master[ds_name]))
            with c2:
                st.pyplot(accuracy_comparison_bar(master[ds_name]))

        # Conclusion
        best_model = max(["tfidf","word2vec","glove"],
            key=lambda m: np.mean([master[ds][m]["accuracy"] for ds in master]))
        st.markdown(f"""
        ### 🏆 Conclusion
        Based on average accuracy across both test datasets,
        **{MODEL_LABELS[best_model]}** achieved the best cross-dataset generalization.
        
        | Method | Pros | Cons |
        |--------|------|------|
        | TF-IDF | Fast, interpretable, strong on clean text | Ignores word order & semantics |
        | Word2Vec | Semantic awareness, handles OOV loosely | Needs large corpus, avg loses word order |
        | GloVe-style | Better rare-word embeddings, wider context | Slower training, similar limits to W2V |
        
        > In practice, TF-IDF often wins on shorter, noisier texts (tweets) while
        > dense embeddings (Word2Vec/GloVe) excel on longer, richer documents (reviews).
        """)

        csv_out = cmp_df.to_csv(index=False)
        st.download_button("⬇️ Download Comparison CSV", csv_out, "cross_dataset_comparison.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Training Summary":
    st.markdown("## 📋 Training Summary")

    results_path = os.path.join(BASE_DIR, "models", "training_results.json")
    if not os.path.exists(results_path):
        st.warning("Training results not found. Run `python train_models.py` first.")
        st.stop()

    with open(results_path) as f:
        results = json.load(f)

    col1, col2, col3 = st.columns(3)
    for col, mname in zip([col1,col2,col3], ["tfidf","word2vec","glove_style"]):
        r = results.get(mname, {})
        col.metric(
            MODEL_LABELS.get(mname, mname),
            f"{r.get('accuracy',0)*100:.2f}%",
            f"Trained in {r.get('train_time','?')}s",
        )

    st.markdown("### Validation Results Table")
    rows = []
    for mname in ["tfidf","word2vec","glove_style"]:
        r = results.get(mname, {})
        rows.append({
            "Model": MODEL_LABELS.get(mname, mname),
            "Accuracy (%)": f"{r.get('accuracy',0)*100:.2f}",
            "Correct": r.get("correct","—"),
            "Incorrect": r.get("incorrect","—"),
            "Train Time (s)": r.get("train_time","—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Raw JSON")
    st.json(results)
