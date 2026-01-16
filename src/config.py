# -*- coding: utf-8 -*-
"""Central config/constants shared by training/export/inference.

Keep these in sync with `pred.py` (inference) to avoid training/inference drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# ---- Dataset columns ----
TARGET_COL = "label"
GROUP_COL = "student_id"

LIKERT_ACADEMIC = "How likely are you to use this model for academic tasks?"
LIKERT_SUBOPT_FREQ = "Based on your experience, how often has this model given you a response that felt suboptimal?"
LIKERT_EXPECT_REF = "How often do you expect this model to provide responses with references or supporting evidence?"
LIKERT_VERIFY_FREQ = "How often do you verify this model's responses?"

ORDINAL_COLS = [
    LIKERT_ACADEMIC,
    LIKERT_SUBOPT_FREQ,
    LIKERT_EXPECT_REF,
    LIKERT_VERIFY_FREQ,
]

MULTI_BEST = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
MULTI_SUBOPT = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

# Multi-select keywords (must match cleaned dataset values)
TASK_KEYWORDS: Dict[str, str] = {
    "math": "Math computations",
    "coding": "Writing or debugging code",
    "data": "Data processing or analysis",
    "draft": "Drafting professional text",
    "writing": "Writing or editing essays",
    "explain": "Explaining complex concepts simply",
    "convert": "Converting content between formats",
}

# ---- Default split sizes ----
DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_SIZE = 0.15

# ---- Softmax(TF-IDF + LR) defaults ----
SOFTMAX_TFIDF_MIN_DF = 2
SOFTMAX_TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)
SOFTMAX_TFIDF_MAX_FEATURES = 20000
SOFTMAX_TFIDF_MAX_DF = 0.95
SOFTMAX_TFIDF_STOP_WORDS = "english"
SOFTMAX_LR_C = 0.2
SOFTMAX_LR_MAX_ITER = 3000

# ---- MLP branch defaults ----
MLP_TEXT_TFIDF_MIN_DF = 2
MLP_TEXT_TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)
MLP_SVD_COMPONENTS = 100

MLP_BATCH_SIZE = 64
MLP_MAX_EPOCHS = 50
MLP_PATIENCE = 6
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_DROPOUT = 0.20
MLP_GRAD_CLIP = 1.0

# ---- Ensemble defaults ----
ENSEMBLE_W_SOFTMAX = 0.5
ENSEMBLE_W_MLP = 0.5


@dataclass(frozen=True)
class Paths:
    """Default paths (overridable via CLI)."""
    data_csv: str = "data/training_data_clean.csv"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
