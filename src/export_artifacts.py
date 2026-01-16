# -*- coding: utf-8 -*-
"""Artifact export utilities.

These artifacts are consumed by `pred.py` (dependency-light inference).
File names are intentionally fixed.

Output files (under artifacts/):
- softmax_vocab.json
- softmax_idf.npy
- softmax_lr.npz
- softmax_config.json
- mlp_preproc.json
- mlp_text_vocab.json
- mlp_text_idf.npy
- mlp_svd.npz
- mlp_maxabs.npy
- mlp_weights.npz
- ensemble.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def export_softmax_artifacts(
    vectorizer,
    clf,
    out_dir: Path,
    *,
    config: Dict,
) -> None:
    """Export TF-IDF vocab/idf + LogisticRegression weights."""
    _ensure_dir(out_dir)

    vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    idf = np.asarray(vectorizer.idf_, dtype=np.float32)

    coef = np.asarray(clf.coef_, dtype=np.float32)
    intercept = np.asarray(clf.intercept_, dtype=np.float32)
    classes = np.asarray(clf.classes_, dtype=object)

    (out_dir / "softmax_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out_dir / "softmax_idf.npy", idf)
    np.savez(out_dir / "softmax_lr.npz", coef=coef, intercept=intercept, classes=classes)
    (out_dir / "softmax_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def export_mlp_artifacts(
    preproc_pipeline,
    torch_model,
    classes: List[str],
    out_dir: Path,
    *,
    text_config: Dict,
) -> None:
    """Export MLP preprocessing + weights for pure-numpy inference."""
    _ensure_dir(out_dir)

    # Pipeline: preprocess (ColumnTransformer) -> scale (MaxAbsScaler)
    ct = preproc_pipeline.named_steps["preprocess"]
    scaler = preproc_pipeline.named_steps["scale"]

    num_cols: List[str] = []
    cat_cols: List[str] = []
    text_cols: List[str] = []

    # Extract fitted transformers and their column lists
    for name, transformer, cols in ct.transformers_:
        if name == "num":
            num_cols = list(cols) if cols is not None else []
        elif name == "cat":
            cat_cols = list(cols) if cols is not None else []
        elif name == "text":
            text_cols = list(cols) if cols is not None else []

    # Numeric medians
    num_medians: List[float] = []
    if num_cols:
        num_pipe = ct.named_transformers_["num"]
        imputer = num_pipe.named_steps["imputer"]
        num_medians = [float(x) for x in imputer.statistics_.tolist()]

    # Categorical categories
    cat_categories: Dict[str, List[str]] = {}
    if cat_cols:
        cat_pipe = ct.named_transformers_["cat"]
        ohe = cat_pipe.named_steps["ohe"]
        # ohe.categories_ is list aligned with cat_cols
        for col, cats in zip(cat_cols, ohe.categories_):
            cat_categories[str(col)] = [str(x) for x in cats.tolist()]

    # Text tfidf + svd (can be absent if text_cols empty)
    text_vocab: Dict[str, int] = {}
    text_idf = np.zeros((0,), dtype=np.float32)
    svd_components = np.zeros((0, 0), dtype=np.float32)
    if text_cols:
        text_pipe = ct.named_transformers_["text"]
        tfidf = text_pipe.named_steps["tfidf"]
        svd = text_pipe.named_steps["svd"]
        text_vocab = {k: int(v) for k, v in tfidf.vocabulary_.items()}
        text_idf = np.asarray(tfidf.idf_, dtype=np.float32)
        svd_components = np.asarray(svd.components_, dtype=np.float32)

    # MaxAbsScaler stats
    maxabs = np.asarray(scaler.max_abs_, dtype=np.float32)

    # Torch model weights -> numpy weights for pred.py
    sd = torch_model.state_dict()
    # Our MLP is defined in src/train.py and uses .net Sequential:
    # 0: Linear(in_dim, 256), 3: Linear(256,128), 6: Linear(128,n_classes)
    W1_t = sd["net.0.weight"].detach().cpu().numpy().astype(np.float32)  # (256, in_dim)
    b1 = sd["net.0.bias"].detach().cpu().numpy().astype(np.float32)
    W2_t = sd["net.3.weight"].detach().cpu().numpy().astype(np.float32)  # (128, 256)
    b2 = sd["net.3.bias"].detach().cpu().numpy().astype(np.float32)
    W3_t = sd["net.6.weight"].detach().cpu().numpy().astype(np.float32)  # (n_classes, 128)
    b3 = sd["net.6.bias"].detach().cpu().numpy().astype(np.float32)

    # Transpose to match pred.py: X @ W + b
    W1 = W1_t.T  # (in_dim, 256)
    W2 = W2_t.T  # (256, 128)
    W3 = W3_t.T  # (128, n_classes)

    # Save JSON preproc meta
    preproc_meta = {
        "num_cols": [str(c) for c in num_cols],
        "num_medians": num_medians,
        "cat_cols": [str(c) for c in cat_cols],
        "cat_categories": cat_categories,
        "text_cols": [str(c) for c in text_cols],
        "text_config": text_config,
    }
    (out_dir / "mlp_preproc.json").write_text(json.dumps(preproc_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save text artifacts (even if empty, keep files present for pred.py)
    (out_dir / "mlp_text_vocab.json").write_text(json.dumps(text_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out_dir / "mlp_text_idf.npy", text_idf)
    np.savez(out_dir / "mlp_svd.npz", components=svd_components)

    # Save scaler + MLP weights
    np.save(out_dir / "mlp_maxabs.npy", maxabs)
    np.savez(
        out_dir / "mlp_weights.npz",
        W1=W1, b1=b1,
        W2=W2, b2=b2,
        W3=W3, b3=b3,
        classes=np.asarray(list(map(str, classes)), dtype=object),
    )


def export_ensemble_weights(out_dir: Path, w_softmax: float, w_mlp: float) -> None:
    _ensure_dir(out_dir)
    (out_dir / "ensemble.json").write_text(
        json.dumps({"w_softmax": float(w_softmax), "w_mlp": float(w_mlp)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
