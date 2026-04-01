"""
column_detector.py
------------------
Intelligent column detection for various dataset types:
  • Supervised datasets (with labels)
  • Unsupervised datasets (text only)
  • Multi-column datasets with different formats
"""

import pandas as pd
from typing import Dict, Optional, List, Tuple


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intelligently detect the text column from a DataFrame.

    Strategy:
    1. Check for common text column names
    2. Find string columns with longest average length
    3. Find columns with high text-like characteristics

    Returns:
        Column name if found, None otherwise
    """
    # Common text column names (case-insensitive)
    text_candidates = [
        "text",
        "sentence",
        "content",
        "review",
        "tweet",
        "message",
        "comment",
        "description",
        "body",
        "post",
        "document",
        "passage",
        "article",
        "headline",
        "title",
        "question",
        "answer",
        "feedback",
    ]

    columns_lower = {col.lower(): col for col in df.columns}

    # Strategy 1: Exact name match
    for candidate in text_candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]

    # Strategy 2: Partial name match
    for candidate in text_candidates:
        for col_lower, col_original in columns_lower.items():
            if candidate in col_lower:
                return col_original

    # Strategy 3: Find string columns with longest average length
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not string_cols:
        return None

    # Calculate average length for each string column
    avg_lengths = {}
    for col in string_cols:
        try:
            # Filter out NaN and calculate average text length
            texts = df[col].dropna().astype(str)
            if len(texts) > 0:
                avg_length = texts.str.len().mean()
                # Prefer columns with reasonable text length (> 20 chars on average)
                if avg_length > 20:
                    avg_lengths[col] = avg_length
        except:
            continue

    if avg_lengths:
        # Return column with longest average text
        return max(avg_lengths, key=avg_lengths.get)

    # Fallback: return first string column
    return string_cols[0] if string_cols else None


def detect_label_column(
    df: pd.DataFrame, exclude_col: Optional[str] = None
) -> Optional[str]:
    """
    Intelligently detect the label/target column from a DataFrame.

    Strategy:
    1. Check for common label column names
    2. Find columns with low cardinality (categorical-like)
    3. Find numeric columns with limited unique values

    Args:
        df: DataFrame to analyze
        exclude_col: Column name to exclude (typically the text column)

    Returns:
        Column name if found, None otherwise (for unsupervised data)
    """
    # Common label column names (case-insensitive)
    label_candidates = [
        "label",
        "target",
        "class",
        "sentiment",
        "category",
        "rating",
        "score",
        "polarity",
        "emotion",
        "intent",
        "tag",
        "type",
        "y",
    ]

    columns_lower = {col.lower(): col for col in df.columns}
    available_cols = [col for col in df.columns if col != exclude_col]

    if not available_cols:
        return None

    # Strategy 1: Exact name match
    for candidate in label_candidates:
        if candidate in columns_lower and columns_lower[candidate] in available_cols:
            return columns_lower[candidate]

    # Strategy 2: Partial name match
    for candidate in label_candidates:
        for col_lower, col_original in columns_lower.items():
            if candidate in col_lower and col_original in available_cols:
                return col_original

    # Strategy 3: Find columns with low cardinality (likely categorical)
    # Only consider columns with 2-50 unique values
    candidates_by_cardinality = []
    for col in available_cols:
        try:
            n_unique = df[col].nunique()
            n_rows = len(df)

            # Good label candidates: 2-50 unique values, and < 20% of total rows
            if 2 <= n_unique <= 50 and n_unique < n_rows * 0.2:
                candidates_by_cardinality.append((col, n_unique))
        except:
            continue

    if candidates_by_cardinality:
        # Prefer columns with fewer unique values
        candidates_by_cardinality.sort(key=lambda x: x[1])
        return candidates_by_cardinality[0][0]

    # No clear label column found (might be unsupervised data)
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect both text and label columns from a DataFrame.

    Returns:
        Dictionary with keys:
        - 'text': detected text column name (or None)
        - 'label': detected label column name (or None for unsupervised)
        - 'is_supervised': bool indicating if labels were found
        - 'other_columns': list of other columns in the dataset
    """
    text_col = detect_text_column(df)
    label_col = detect_label_column(df, exclude_col=text_col)

    other_cols = [col for col in df.columns if col not in [text_col, label_col]]

    return {
        "text": text_col,
        "label": label_col,
        "is_supervised": label_col is not None,
        "other_columns": other_cols,
    }


def get_column_info(df: pd.DataFrame, col: str) -> Dict:
    """Get detailed information about a column."""
    info = {
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": df[col].nunique(),
        "n_null": df[col].isna().sum(),
        "sample_values": df[col].dropna().head(3).tolist(),
    }

    # Add text-specific info for object columns
    if df[col].dtype == "object":
        try:
            texts = df[col].dropna().astype(str)
            info["avg_length"] = round(texts.str.len().mean(), 1)
            info["max_length"] = texts.str.len().max()
        except:
            pass

    return info


def validate_dataset(
    df: pd.DataFrame, require_labels: bool = False
) -> Tuple[bool, str, Dict]:
    """
    Validate a dataset and provide detailed feedback.

    Args:
        df: DataFrame to validate
        require_labels: If True, labels are required

    Returns:
        (is_valid, message, detected_columns)
    """
    if df.empty:
        return False, "❌ Dataset is empty", {}

    detected = detect_columns(df)

    if not detected["text"]:
        return (
            False,
            f"❌ Could not detect text column. Found columns: {list(df.columns)}",
            detected,
        )

    if require_labels and not detected["is_supervised"]:
        return (
            False,
            f"❌ No label column detected. This appears to be unsupervised data. Found columns: {list(df.columns)}",
            detected,
        )

    # Success message
    if detected["is_supervised"]:
        msg = f"✅ Detected text column: **{detected['text']}**, label column: **{detected['label']}**"
    else:
        msg = f"✅ Detected text column: **{detected['text']}** (unsupervised data - no labels)"

    if detected["other_columns"]:
        msg += f"\n\n📋 Other columns: {', '.join(detected['other_columns'])}"

    return True, msg, detected
