# -*- coding: utf-8 -*-
"""Feature engineering helpers shared by training & inference.

Key goals:
- Avoid data leakage (drop ID-like columns; group-aware split).
- Keep training/inference text construction consistent for the Softmax(TF-IDF+LR) branch.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import pandas as pd

from . import config


_ID_LIKE_RE = re.compile(r"(?:^|[_\-])(?:id|uuid|guid)(?:$|[_\-])", re.I)


def detect_id_like_columns(columns: Iterable[str]) -> List[str]:
    """Heuristically detect ID-like columns by name."""
    return [c for c in columns if _ID_LIKE_RE.search(str(c))]


def detect_text_like_columns(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    """Heuristic: free-text columns tend to have longer avg length or higher uniqueness."""
    text_cols: List[str] = []
    for c in candidate_cols:
        s = df[c].astype(str)
        if s.str.len().mean() >= 30 or (s.nunique(dropna=True) / max(1, len(s)) >= 0.2):
            text_cols.append(c)
    return text_cols


def join_text_columns(X) -> pd.Series:
    """Join multiple text columns into a single string per row."""
    if isinstance(X, pd.DataFrame):
        return X.astype(str).fillna("").agg(" ".join, axis=1)
    X = pd.DataFrame(X)
    return X.astype(str).fillna("").agg(" ".join, axis=1)


def expand_multi_select(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Split a multi-select text column into multiple 0/1 indicator columns.

    Example output columns:
      - best_math, best_coding, ...
    The original multi-select column is dropped.
    """
    if col not in df.columns:
        return df
    s = df[col].fillna("")
    for key, pat in config.TASK_KEYWORDS.items():
        new_col = f"{prefix}_{key}"
        df[new_col] = s.astype(str).str.contains(pat, regex=False).astype("float32")
    return df.drop(columns=[col])


def build_text_concat_for_softmax(df: pd.DataFrame) -> pd.Series:
    """Build per-row text for the Softmax(TF-IDF+LR) branch.

    This function is intentionally aligned with `pred.py` to keep training/inference consistent:
    - Likert columns: "col: value"
    - Multi-select original text: "col: raw_value" (if present)
    - Multi-select expanded tokens: "Selected_best_math" etc.
    - Other columns (excluding label / student_id / id-like / likert / multi-select): appended as text.

    Returns:
        pd.Series of concatenated text (len == len(df))
    """
    dfp = df.copy()

    # multi-select expansion (for per-row tokens)
    dfp = expand_multi_select(dfp, config.MULTI_BEST, prefix="best")
    dfp = expand_multi_select(dfp, config.MULTI_SUBOPT, prefix="subopt")

    parts: List[pd.Series] = []

    # Likert (raw)
    for col in config.ORDINAL_COLS:
        if col in df.columns:
            parts.append((col + ": " + df[col].astype(str).fillna("")))

    # Multi-select original raw text (optional, but kept consistent with pred.py)
    for col in [config.MULTI_BEST, config.MULTI_SUBOPT]:
        if col in df.columns:
            parts.append((col + ": " + df[col].fillna("").astype(str)))

    # multi-hot tokens (row-wise)
    for prefix in ["best_", "subopt_"]:
        pref_cols = [c for c in dfp.columns if c.startswith(prefix)]
        if pref_cols:
            toks = dfp[pref_cols].apply(
                lambda r: " ".join([f"Selected_{c}" for c, v in r.items() if float(v) == 1.0]),
                axis=1
            ).astype(str)
            parts.append(toks)

    # Other cols as additional text
    processed = set(config.ORDINAL_COLS + [config.MULTI_BEST, config.MULTI_SUBOPT])
    drop_cols = set([config.GROUP_COL, config.TARGET_COL]) | processed | set(detect_id_like_columns(df.columns))
    other_cols = [c for c in df.columns if c not in drop_cols]
    if other_cols:
        others = df[other_cols].astype(str).fillna("").agg(" [SEP] ".join, axis=1)
        parts.append(others)

    if not parts:
        return pd.Series([""] * len(df), index=df.index)

    final = parts[0].astype(str)
    for p in parts[1:]:
        final = final + " [SEP] " + p.astype(str)
    return final
