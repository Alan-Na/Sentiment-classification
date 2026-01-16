# -*- coding: utf-8 -*-
"""Train + evaluate + export artifacts + generate plots.

Run from repo root (recommended):

    python -m src.train --data data/training_data_clean.csv --seed 42

Outputs:
- artifacts/*  (for pred.py)
- reports/metrics.json
- reports/confusion_matrix.png
- reports/per_class_f1.png
"""


from __future__ import annotations

# Allow running as a script: `python src/train.py`
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))



import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

from src import config, features, reporting
from src.export_artifacts import (
    export_softmax_artifacts,
    export_mlp_artifacts,
    export_ensemble_weights,
)


# ---------------- Reproducibility ----------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Data ----------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {config.TARGET_COL}")
    if config.GROUP_COL not in df.columns:
        df[config.GROUP_COL] = np.arange(len(df))
        print("[WARN] student_id missing; falling back to row index as group.")
    return df


def grouped_split_with_all_classes(
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    seed: int,
    max_tries: int = 50,
):
    """GroupShuffleSplit with retry to cover all classes on both sides."""
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
    print("[WARN] Split retry exceeded; using the last split (may miss some classes).")

    return last


def align_proba(P_raw: np.ndarray, model_classes: List[str], classes: List[str]) -> np.ndarray:
    """Align probability columns to a reference classes order."""
    col_map = {str(c): i for i, c in enumerate(model_classes)}
    P = np.zeros((P_raw.shape[0], len(classes)), dtype=np.float32)
    for j, c in enumerate(classes):
        if c in col_map:
            P[:, j] = P_raw[:, col_map[c]]
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return P / row_sum


# ---------------- Softmax branch ----------------
def train_softmax_branch(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> Tuple[TfidfVectorizer, LogisticRegression, np.ndarray, np.ndarray, List[str]]:
    """TF-IDF + multinomial LogisticRegression."""
    y = df[config.TARGET_COL].astype(str)
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    texts = features.build_text_concat_for_softmax(df)
    X_tr_text = texts.iloc[train_idx].astype(str).tolist()
    X_te_text = texts.iloc[test_idx].astype(str).tolist()

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=config.SOFTMAX_TFIDF_NGRAM_RANGE,
        min_df=config.SOFTMAX_TFIDF_MIN_DF,
        max_df=config.SOFTMAX_TFIDF_MAX_DF,
        stop_words=config.SOFTMAX_TFIDF_STOP_WORDS,
        max_features=config.SOFTMAX_TFIDF_MAX_FEATURES,
        dtype=np.float32,
    )
    Xtr = vec.fit_transform(X_tr_text)
    Xte = vec.transform(X_te_text)

    clf = LogisticRegression(
        solver="lbfgs",
        C=config.SOFTMAX_LR_C,
        max_iter=config.SOFTMAX_LR_MAX_ITER,
        random_state=seed,
    )
    clf.fit(Xtr, y_tr)

    P_tr_raw = clf.predict_proba(Xtr)
    P_te_raw = clf.predict_proba(Xte)

    classes = sorted(y_tr.unique().tolist())
    P_tr = align_proba(P_tr_raw, clf.classes_.tolist(), classes)
    P_te = align_proba(P_te_raw, clf.classes_.tolist(), classes)

    return vec, clf, P_tr, P_te, classes


# ---------------- MLP branch ----------------
class NPDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, p: float = config.MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def eval_torch_probs(model: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    loader = DataLoader(NPDataset(X, np.zeros((len(X),), dtype=np.int64)), batch_size=batch_size, shuffle=False)
    probs_all = []
    for xb, _ in loader:
        xb = xb.to(device=device, dtype=torch.float32)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.detach().cpu().numpy())
    return np.vstack(probs_all) if probs_all else np.zeros((0, 0), dtype=np.float32)


def build_mlp_preprocessor(
    X_train: pd.DataFrame,
    *,
    seed: int,
    svd_components: int,
) -> Pipeline:
    """Build preprocessing pipeline for the MLP branch."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in X_train.columns if c not in num_cols]
    text_cols = features.detect_text_like_columns(X_train, obj_cols)
    cat_cols = [c for c in obj_cols if c not in text_cols]

    # numeric
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # categorical
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", ohe),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if text_cols:
        text_pipe = Pipeline([
            ("join", FunctionTransformer(features.join_text_columns, validate=False)),
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                token_pattern=r"(?u)\b\w\w+\b",
                ngram_range=config.MLP_TEXT_TFIDF_NGRAM_RANGE,
                min_df=config.MLP_TEXT_TFIDF_MIN_DF,
                dtype=np.float32,
            )),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=seed)),
        ])
        transformers.append(("text", text_pipe, text_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,  # force dense
    )

    return Pipeline([
        ("preprocess", ct),
        ("scale", MaxAbsScaler()),
    ])


def fit_mlp_branch(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    classes: List[str],
    seed: int,
    device: str,
) -> Tuple[Pipeline, MLP, np.ndarray, np.ndarray]:
    """Fit MLP branch and return (preproc, model, P_train, P_test)."""
    seed_all(seed)

    # Prepare features
    dfm = df.copy()

    # Drop ID-like cols (incl. student_id) from feature side
    id_like = features.detect_id_like_columns([c for c in dfm.columns if c != config.TARGET_COL])
    if id_like:
        dfm = dfm.drop(columns=id_like, errors="ignore")

    # Expand multi-select into numeric 0/1, drop original text cols
    dfm = features.expand_multi_select(dfm, config.MULTI_BEST, prefix="best")
    dfm = features.expand_multi_select(dfm, config.MULTI_SUBOPT, prefix="subopt")

    X = dfm.drop(columns=[config.TARGET_COL])
    y = dfm[config.TARGET_COL].astype(str)

    X_tr, X_val, X_te = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_tr, y_val, y_te = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

    # Build preprocessor using *train subset only* (avoid leakage)
    svd_components = config.MLP_SVD_COMPONENTS
    pre = build_mlp_preprocessor(X_tr, seed=seed, svd_components=svd_components)

    # Robust SVD components (if n_components too large, rebuild smaller)
    try:
        pre.fit(X_tr, y_tr)
    except ValueError as e:
        msg = str(e)
        if "n_components" in msg and "must be" in msg:
            # Estimate an upper bound
            # We re-fit a tfidf alone to know vocab size
            tmp = build_mlp_preprocessor(X_tr, seed=seed, svd_components=1)
            tmp.fit(X_tr, y_tr)
            ct = tmp.named_steps["preprocess"]
            # If text transformer exists, fetch tfidf vocab size
            n_features = None
            for name, trans, cols in ct.transformers_:
                if name == "text":
                    tfidf = trans.named_steps["tfidf"]
                    n_features = len(tfidf.vocabulary_)
            n_samples = len(X_tr)
            if n_features is None:
                raise
            new_k = max(1, min(config.MLP_SVD_COMPONENTS, n_samples - 1, n_features - 1))
            print(f"[WARN] SVD n_components too large; retry with n_components={new_k}")
            pre = build_mlp_preprocessor(X_tr, seed=seed, svd_components=new_k)
            pre.fit(X_tr, y_tr)
        else:
            raise

    Xtr_np = np.asarray(pre.transform(X_tr), dtype=np.float32)
    Xval_np = np.asarray(pre.transform(X_val), dtype=np.float32)
    Xte_np = np.asarray(pre.transform(X_te), dtype=np.float32)

    cls2idx = {c: i for i, c in enumerate(classes)}
    ytr_np = np.array([cls2idx[s] for s in y_tr], dtype=np.int64)
    yval_np = np.array([cls2idx[s] for s in y_val], dtype=np.int64)

    model = MLP(in_dim=Xtr_np.shape[1], n_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.MLP_LR, weight_decay=config.MLP_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(NPDataset(Xtr_np, ytr_np), batch_size=config.MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NPDataset(Xval_np, yval_np), batch_size=config.MLP_BATCH_SIZE, shuffle=False)

    best_val = math.inf
    best_state = None
    patience_left = config.MLP_PATIENCE

    for _epoch in range(1, config.MLP_MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.MLP_GRAD_CLIP)
            opt.step()

        # val loss
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            n = 0
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)
                l = criterion(model(xb), yb).item()
                val_loss_sum += l * len(yb)
                n += len(yb)
            val_loss = val_loss_sum / max(1, n)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.MLP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # probs on train/test
    P_tr = eval_torch_probs(model, Xtr_np, batch_size=config.MLP_BATCH_SIZE, device=device)
    P_te = eval_torch_probs(model, Xte_np, batch_size=config.MLP_BATCH_SIZE, device=device)

    # Ensure float32
    P_tr = np.asarray(P_tr, dtype=np.float32)
    P_te = np.asarray(P_te, dtype=np.float32)

    return pre, model, P_tr, P_te


def probs_to_preds(P: np.ndarray, classes: List[str]) -> List[str]:
    idx = np.argmax(P, axis=1)
    return [classes[i] for i in idx]


def save_metrics_markdown(all_metrics: Dict, out_path: Path) -> None:
    lines = []
    lines.append("# Metrics summary\n")
    lines.append("| Model | Accuracy | Macro-F1 | Weighted-F1 |\n")
    lines.append("|------|----------|----------|-------------|\n")
    for key in ["knn_baseline", "softmax", "mlp", "ensemble"]:
        if key not in all_metrics:
            continue
        m = all_metrics[key]
        lines.append(
            f"| {key} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |\n"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=config.Paths().data_csv)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=config.DEFAULT_TEST_SIZE)
    parser.add_argument("--val_size", type=float, default=config.DEFAULT_VAL_SIZE)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no_export", action="store_true", help="Only train/eval; do not write artifacts/")
    parser.add_argument("--no_baseline", action="store_true", help="Skip KNN baseline")
    parser.add_argument("--artifacts_dir", type=str, default=config.Paths().artifacts_dir)
    parser.add_argument("--reports_dir", type=str, default=config.Paths().reports_dir)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_all(args.seed)

    df = load_df(args.data)
    y_all = df[config.TARGET_COL].astype(str)
    groups = df[config.GROUP_COL].astype(str)

    tr_idx, te_idx = grouped_split_with_all_classes(y_all, groups, args.test_size, args.seed)
    y_tr_all = y_all.iloc[tr_idx]
    groups_tr = groups.iloc[tr_idx]

    # train/val inside train
    tr_in_idx, val_in_idx = grouped_split_with_all_classes(y_tr_all, groups_tr, args.val_size, args.seed + 1)
    train_idx = np.array(tr_idx)[tr_in_idx]
    val_idx = np.array(tr_idx)[val_in_idx]
    test_idx = np.array(te_idx)

    classes = sorted(y_all.iloc[train_idx].unique().tolist())

    # ---- Train branches ----
    vec, lr_clf, P_lr_tr, P_lr_te, _classes_softmax = train_softmax_branch(df, train_idx, test_idx, seed=args.seed)

    preproc, mlp_model, P_mlp_tr, P_mlp_te = fit_mlp_branch(
        df, train_idx, val_idx, test_idx, classes=classes, seed=args.seed, device=device
    )

    # ---- Ensemble ----
    w_softmax = float(config.ENSEMBLE_W_SOFTMAX)
    w_mlp = float(config.ENSEMBLE_W_MLP)
    P_ens_tr = w_softmax * P_lr_tr + w_mlp * P_mlp_tr
    P_ens_te = w_softmax * P_lr_te + w_mlp * P_mlp_te
    # normalize
    P_ens_tr = P_ens_tr / np.clip(P_ens_tr.sum(axis=1, keepdims=True), 1e-12, None)
    P_ens_te = P_ens_te / np.clip(P_ens_te.sum(axis=1, keepdims=True), 1e-12, None)

    # ---- Predictions ----
    y_train_true = y_all.iloc[train_idx].tolist()
    y_test_true = y_all.iloc[test_idx].tolist()

    pred_lr = probs_to_preds(P_lr_te, classes)
    pred_mlp = probs_to_preds(P_mlp_te, classes)
    pred_ens = probs_to_preds(P_ens_te, classes)

    # ---- Metrics + plots ----
    out_reports = Path(args.reports_dir)
    out_reports.mkdir(parents=True, exist_ok=True)

    metrics_softmax = reporting.compute_metrics(y_test_true, pred_lr, labels=classes)
    metrics_mlp = reporting.compute_metrics(y_test_true, pred_mlp, labels=classes)
    metrics_ens = reporting.compute_metrics(y_test_true, pred_ens, labels=classes)

    all_metrics: Dict[str, Dict] = {
        "softmax": metrics_softmax,
        "mlp": metrics_mlp,
        "ensemble": metrics_ens,
        "split": {
            "seed": args.seed,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_classes": int(len(classes)),
        },
    }

    # Optional baseline on same split
    if not args.no_baseline:
        try:
            from baselines.knn_baseline import build_baseline_features, impute_train_median
            from sklearn.neighbors import KNeighborsClassifier

            Xb = build_baseline_features(df)
            Xb_tr, Xb_te = Xb[train_idx], Xb[test_idx]
            Xb_tr, Xb_te = impute_train_median(Xb_tr, Xb_te)
            yb_tr = np.array(y_train_true, dtype=object)
            yb_te = np.array(y_test_true, dtype=object)

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(Xb_tr, yb_tr)
            pred_knn = knn.predict(Xb_te).tolist()
            all_metrics["knn_baseline"] = reporting.compute_metrics(y_test_true, pred_knn, labels=classes)
        except Exception as e:
            print("[WARN] Baseline failed:", repr(e))

    reporting.save_json(all_metrics, out_reports / "metrics.json")
    save_metrics_markdown(
        {
            k: v for k, v in all_metrics.items()
            if k in {"knn_baseline", "softmax", "mlp", "ensemble"}
        },
        out_reports / "metrics.md",
    )

    # plots for ensemble (main model)
    reporting.plot_confusion_matrix(
        np.asarray(metrics_ens["confusion_matrix"], dtype=int),
        labels=classes,
        out_path=out_reports / "confusion_matrix.png",
        title="Ensemble – Confusion Matrix",
    )
    reporting.plot_per_class_f1(
        metrics_ens,
        out_path=out_reports / "per_class_f1.png",
        title="Ensemble – Per-class F1",
    )

    print("=== Done ===")
    print("Metrics saved to:", str(out_reports / "metrics.json"))
    print("Plots saved to:", str(out_reports / "confusion_matrix.png"), "and", str(out_reports / "per_class_f1.png"))

    # ---- Export artifacts ----
    if not args.no_export:
        out_art = Path(args.artifacts_dir)
        out_art.mkdir(parents=True, exist_ok=True)

        export_softmax_artifacts(
            vec,
            lr_clf,
            out_art,
            config={
                "ngram_range": list(config.SOFTMAX_TFIDF_NGRAM_RANGE),
                "lowercase": True,
                "strip_accents": "unicode",
                "token_pattern": r"(?u)\\b\\w\\w+\\b",
                "min_df": config.SOFTMAX_TFIDF_MIN_DF,
                "max_df": config.SOFTMAX_TFIDF_MAX_DF,
                "stop_words": config.SOFTMAX_TFIDF_STOP_WORDS,
                "max_features": config.SOFTMAX_TFIDF_MAX_FEATURES,
            },
        )

        export_mlp_artifacts(
            preproc,
            mlp_model,
            classes,
            out_art,
            text_config={
                "ngram_range": list(config.MLP_TEXT_TFIDF_NGRAM_RANGE),
                "lowercase": True,
                "strip_accents": "unicode",
                "token_pattern": r"(?u)\\b\\w\\w+\\b",
                "min_df": config.MLP_TEXT_TFIDF_MIN_DF,
            },
        )

        export_ensemble_weights(out_art, w_softmax=w_softmax, w_mlp=w_mlp)
        print("Artifacts saved to:", str(out_art.resolve()))
    else:
        print("[INFO] --no_export set; artifacts not written.")


if __name__ == "__main__":
    main()
