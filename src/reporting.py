# -*- coding: utf-8 -*-
"""Reporting utilities: metrics + plots (confusion matrix, per-class F1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """Return metrics dict suitable for JSON serialization."""
    y_true = list(map(str, y_true))
    y_pred = list(map(str, y_pred))

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    weighted_f1 = float(np.average(f1, weights=sup)) if sup.sum() else 0.0

    per_class = {}
    for i, lab in enumerate(labels):
        per_class[lab] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path, title: str) -> None:
    """Save a confusion matrix plot (matplotlib only)."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), max(5, 0.6 * len(labels))))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks, labels=labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks, labels=labels)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # annotate counts
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(metrics: Dict, out_path: Path, title: str) -> None:
    labels = metrics.get("labels", [])
    per_class = metrics.get("per_class", {})
    f1s = [per_class.get(l, {}).get("f1", 0.0) for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4))
    ax.bar(labels, f1s)
    ax.set_title(title)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
