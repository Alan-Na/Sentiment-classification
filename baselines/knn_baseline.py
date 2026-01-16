# -*- coding: utf-8 -*-
"""Simple KNN baseline for comparison (not meant to be SOTA).

Features:
- Likert ratings -> numeric (extract leading digit)
- Multi-select columns -> binary indicators per task keyword

Split:
- GroupShuffleSplit on student_id (if available) with retry to cover all classes.
"""


from __future__ import annotations

# Allow running as a script: `python baselines/knn_baseline.py`
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))



import argparse
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from src import config  # repo-root import


_RATING_RE = re.compile(r"^(\d+)")


def _extract_rating(x) -> float:
    if pd.isna(x):
        return np.nan
    m = _RATING_RE.match(str(x).strip())
    return float(m.group(1)) if m else np.nan


def grouped_split_with_all_classes(y: pd.Series, groups: pd.Series, test_size: float, seed: int, max_tries: int = 50):
    all_classes = np.unique(y.astype(str))
    rng = np.random.RandomState(seed)
    last = None
    for _ in range(max_tries):
        rs = int(rng.randint(0, 2**31 - 1))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr_idx, te_idx = next(gss.split(np.zeros(len(y)), y, groups))
        last = (tr_idx, te_idx)
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        if (len(np.unique(y_tr)) == len(all_classes)) and (len(np.unique(y_te)) == len(all_classes)):
            return tr_idx, te_idx
    return last


def build_baseline_features(df: pd.DataFrame) -> np.ndarray:
    """Return baseline feature matrix."""
    feats = []

    # Likert numeric
    for col in config.ORDINAL_COLS:
        if col in df.columns:
            feats.append(df[col].apply(_extract_rating).to_numpy(dtype=np.float32))
        else:
            feats.append(np.full((len(df),), np.nan, dtype=np.float32))

    # Multi-select indicators (best/subopt)
    for col, prefix in [(config.MULTI_BEST, "best"), (config.MULTI_SUBOPT, "subopt")]:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
        else:
            s = pd.Series([""] * len(df))
        for key, pat in config.TASK_KEYWORDS.items():
            feats.append(s.str.contains(pat, regex=False).astype("float32").to_numpy())

    X = np.vstack(feats).T  # (n, d)
    return X


def impute_train_median(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    med = np.nanmedian(X_train, axis=0)
    # if a column is all-NaN in train, nanmedian returns NaN -> replace by 0.0
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    # impute
    inds_tr = np.where(np.isnan(X_train))
    X_train[inds_tr] = np.take(med, inds_tr[1])
    inds_te = np.where(np.isnan(X_test))
    X_test[inds_te] = np.take(med, inds_te[1])
    return X_train, X_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=config.Paths().data_csv)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=config.DEFAULT_TEST_SIZE)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {config.TARGET_COL}")

    if config.GROUP_COL not in df.columns:
        df[config.GROUP_COL] = np.arange(len(df))
        print("[WARN] student_id missing; falling back to row index as group.")

    y = df[config.TARGET_COL].astype(str)
    groups = df[config.GROUP_COL].astype(str)

    tr_idx, te_idx = grouped_split_with_all_classes(y, groups, args.test_size, args.seed)
    if tr_idx is None or te_idx is None:
        raise RuntimeError("Failed to split data.")

    X = build_baseline_features(df)
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y.iloc[tr_idx].to_numpy(), y.iloc[te_idx].to_numpy()

    X_train, X_test = impute_train_median(X_train, X_test)

    clf = KNeighborsClassifier(n_neighbors=args.k)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    # Metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = accuracy_score(y_test, pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, pred, labels=sorted(y.unique()), zero_division=0)
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0

    print("=== KNN Baseline ===")
    print(f"Test accuracy : {acc:.4f}")
    print(f"Test macro-F1 : {macro_f1:.4f}")
    print("Confusion matrix shape:", confusion_matrix(y_test, pred, labels=sorted(y.unique())).shape)


if __name__ == "__main__":
    main()
