# Python 3.10+

from __future__ import annotations
import os
import re
import json
import math
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# 配置：工件(artifacts)文件路径
# ------------------------------
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Softmax(TF-IDF + LR) 分支
SOFTMAX_VOCAB_JSON = ARTIFACT_DIR / "softmax_vocab.json"    # {"token": index, ...}
SOFTMAX_IDF_NPY     = ARTIFACT_DIR / "softmax_idf.npy"       # shape=(V,)
SOFTMAX_LR_NPZ      = ARTIFACT_DIR / "softmax_lr.npz"        # keys: coef, intercept, classes
SOFTMAX_CFG_JSON    = ARTIFACT_DIR / "softmax_config.json"   # {"ngram_range":[1,2], "lowercase":true, "strip_accents":"unicode"}

# MLP(TF-IDF→SVD→MLP) 分支
MLP_PREPROC_JSON   = ARTIFACT_DIR / "mlp_preproc.json"       # 列信息与填充值、OHE 类别、文本设置
MLP_TEXT_VOCAB_JSON= ARTIFACT_DIR / "mlp_text_vocab.json"    # 文本词表
MLP_TEXT_IDF_NPY   = ARTIFACT_DIR / "mlp_text_idf.npy"       # 文本 idf
MLP_SVD_NPZ        = ARTIFACT_DIR / "mlp_svd.npz"            # keys: components  (n_comp x V_text)
MLP_MAXABS_NPY     = ARTIFACT_DIR / "mlp_maxabs.npy"         # MaxAbsScaler 的 max_abs 向量
MLP_WEIGHTS_NPZ    = ARTIFACT_DIR / "mlp_weights.npz"        # keys: W1,b1,W2,b2,W3,b3 (见下)

# 集成权重（可选；不存在则默认 0.5/0.5）
ENSEMBLE_JSON      = ARTIFACT_DIR / "ensemble.json"          # {"w_softmax":0.5,"w_mlp":0.5}


# ------------------------------
# 与训练代码一致地列名/规则（保持同步）
# ------------------------------
TARGET_COL   = "label"   # 仅用于文档完整性；推理不读取标签
STUDENT_ID   = "student_id"

# Likert & 多选列（名称需与训练时一致）
LIKERT_ACADEMIC   = "How likely are you to use this model for academic tasks?"
LIKERT_SUBOPT_FREQ= "Based on your experience, how often has this model given you a response that felt suboptimal?"
LIKERT_EXPECT_REF = "How often do you expect this model to provide responses with references or supporting evidence?"
LIKERT_VERIFY_FREQ= "How often do you verify this model's responses?"
ORDINAL_COLS = [LIKERT_ACADEMIC, LIKERT_SUBOPT_FREQ, LIKERT_EXPECT_REF, LIKERT_VERIFY_FREQ]

MULTI_BEST   = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
MULTI_SUBOPT = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

# 训练时用于识别多选题关键字的子串（需保持一致）
TASK_KEYWORDS = {
    "math": "Math computations",
    "coding": "Writing or debugging code",
    "data": "Data processing or analysis",
    "draft": "Drafting professional text",
    "writing": "Writing or editing essays",
    "explain": "Explaining complex concepts simply",
    "convert": "Converting content between formats",
}

# ------------------------------
# 通用工具
# ------------------------------
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

def strip_accents_unicode(s: str) -> str:
    try:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    except Exception:
        pass
    return s

def detect_id_like_columns(columns: List[str]) -> List[str]:
    pat = re.compile(r"(?:^|[_\-])(?:id|uuid|guid)(?:$|[_\-])", re.I)
    return [c for c in columns if pat.search(str(c))]

def ensure_series(x, index):
    if isinstance(x, pd.Series):
        return x
    return pd.Series([str(x)] * len(index), index=index)

# ------------------------------
# 文本分析与 TF-IDF（无 sklearn 版）
# ------------------------------
class TfidfLite:
    """
    仅做与训练时相容的推理：使用训练导出的 vocabulary + idf。
    不在此处执行 min_df/max_df/停用词过滤，因为训练时已经在词表中完成了筛选。
    """
    def __init__(self, vocab: Dict[str, int], idf: np.ndarray,
                 ngram_range: Tuple[int, int] = (1, 2),
                 lowercase: bool = True, strip_accents: bool = True):
        self.vocab = vocab
        self.idf = np.asarray(idf, dtype=np.float32)
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.vocab_size = int(max(vocab.values())) + 1 if vocab else 0

    def _analyze(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = strip_accents_unicode(text)
        tokens = TOKEN_RE.findall(text)
        # ngram: (1,2) 默认
        n1, n2 = self.ngram_range
        grams = []
        if n1 <= 1 <= n2:
            grams.extend(tokens)
        if n2 >= 2:
            grams.extend([" ".join(pair) for pair in zip(tokens, tokens[1:])])
        # 可扩展更高 n 的情形
        return grams

    def transform(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        V = self.vocab_size
        X = np.zeros((n, V), dtype=np.float32)
        for i, t in enumerate(texts):
            counts: Dict[int, float] = {}
            for tok in self._analyze(t):
                j = self.vocab.get(tok)
                if j is not None:
                    counts[j] = counts.get(j, 0.0) + 1.0
            if not counts:
                continue
            row = np.zeros(V, dtype=np.float32)
            for j, tf in counts.items():
                row[j] = tf * self.idf[j]
            # L2 归一化（与 sklearn TfidfVectorizer 默认一致）
            norm = float(np.sqrt((row * row).sum()))
            if norm > 0.0:
                row /= norm
            X[i] = row
        return X


# ------------------------------
# 逻辑回归 Softmax（无 sklearn 版）
# ------------------------------
class SoftmaxLR:
    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: List[str]):
        # coef: (n_classes, V), intercept: (n_classes,)
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.classes = list(map(str, classes))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # logits = X @ W^T + b
        logits = X @ self.coef.T + self.intercept[None, :]
        # 数值稳定的 softmax
        m = logits.max(axis=1, keepdims=True)
        exps = np.exp(logits - m)
        P = exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-12, None)
        return P


# ------------------------------
# SVD 投影（无 sklearn 版）
# ------------------------------
class TruncatedSVDLite:
    def __init__(self, components: np.ndarray):
        # components: (n_comp, V_text)
        self.components = np.asarray(components, dtype=np.float32)

    def transform(self, X_tfidf: np.ndarray) -> np.ndarray:
        # 与 sklearn 的 .transform 一致：X @ components.T
        return X_tfidf @ self.components.T


# ------------------------------
# MLP 前处理（Imputer / OHE / MaxAbs）
# ------------------------------
def impute_numeric(df: pd.DataFrame, num_cols: List[str], medians: List[float]) -> np.ndarray:
    if not num_cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    out = []
    for col, med in zip(num_cols, medians):
        colv = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan] * len(df))
        out.append(colv.fillna(med).to_numpy(dtype=np.float32))
    return np.vstack(out).T  # (n, d_num)

def ohe_categoricals(df: pd.DataFrame, cat_cols: List[str], cat_categories: Dict[str, List[str]]) -> np.ndarray:
    if not cat_cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    blocks = []
    for col in cat_cols:
        cats = cat_categories.get(col, [])
        m = np.zeros((len(df), len(cats)), dtype=np.float32)
        if col in df.columns:
            vals = df[col].astype(str).fillna("missing").to_list()
        else:
            vals = ["missing"] * len(df)
        idx_map = {c: i for i, c in enumerate(cats)}
        for r, v in enumerate(vals):
            j = idx_map.get(v)
            if j is not None:
                m[r, j] = 1.0
        blocks.append(m)
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), dtype=np.float32)

def maxabs_scale(X: np.ndarray, maxabs: np.ndarray) -> np.ndarray:
    maxabs = np.asarray(maxabs, dtype=np.float32)
    X = X.astype(np.float32, copy=True)
    mask = maxabs > 0.0
    X[:, mask] = X[:, mask] / maxabs[mask]
    return X


# ------------------------------
# MLP 前向（无 torch 版）
# ------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)

class MLPInfer:
    def __init__(self, W1, b1, W2, b2, W3, b3, classes: List[str]):
        self.W1 = np.asarray(W1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.W2 = np.asarray(W2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)
        self.W3 = np.asarray(W3, dtype=np.float32)
        self.b3 = np.asarray(b3, dtype=np.float32)
        self.classes = list(map(str, classes))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h1 = relu(X @ self.W1 + self.b1[None, :])
        h2 = relu(h1 @ self.W2 + self.b2[None, :])
        logits = h2 @ self.W3 + self.b3[None, :]
        m = logits.max(axis=1, keepdims=True)
        exps = np.exp(logits - m)
        P = exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-12, None)
        return P


# ------------------------------
# 与训练一致的特征构造（避免信息泄漏）
# ------------------------------
def expand_multi_select(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """把多选题（逗号分隔文本）拆成若干 0/1 数值列，并删除原始长文本列。"""
    if col not in df.columns:
        return df
    s = df[col].fillna("")
    for key, pat in TASK_KEYWORDS.items():
        new_col = f"{prefix}_{key}"
        df[new_col] = s.astype(str).str.contains(pat, regex=False).astype("float32")
    df = df.drop(columns=[col])
    return df

def build_text_concat_for_softmax(df: pd.DataFrame) -> pd.Series:
    """
    Softmax 分支文本：与训练端一致（逐行构造，不使用全表统计）。
    - Likert 保持 "列名: 值" 文本；
    - 多选题原文 + multi-hot 逐行 token；
    - 其他列（去除 id/label/Likert/多选）作为补充文本。
    """
    dfp = df.copy()

    # 多选题展开（用于逐行 token）
    dfp = expand_multi_select(dfp, MULTI_BEST,  prefix="best")
    dfp = expand_multi_select(dfp, MULTI_SUBOPT, prefix="subopt")

    parts = []

    # Likert 原样
    for col in ORDINAL_COLS:
        if col in df.columns:
            parts.append((col + ": " + df[col].astype(str).fillna("")))

    # 多选题原文
    for col in [MULTI_BEST, MULTI_SUBOPT]:
        if col in df.columns:
            parts.append((col + ": " + df[col].fillna("").astype(str)))

    # multi-hot 逐行 token（关键：逐行；不广播）
    for prefix in ["best_", "subopt_"]:
        pref_cols = [c for c in dfp.columns if c.startswith(prefix)]
        if pref_cols:
            toks = dfp[pref_cols].apply(
                lambda r: " ".join([f"Selected_{c}" for c, v in r.items() if float(v) == 1.0]),
                axis=1
            ).astype(str)
            parts.append(toks)

    # 其他列文本（去掉 ID-like / label / Likert / 多选）
    processed = set(ORDINAL_COLS + [MULTI_BEST, MULTI_SUBOPT])
    drop_cols = set([STUDENT_ID, TARGET_COL]) | processed | set(detect_id_like_columns(df.columns))
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


# ------------------------------
# 载入工件（权重/词表/配置）
# ------------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_softmax_artifacts():
    if not (SOFTMAX_VOCAB_JSON.exists() and SOFTMAX_IDF_NPY.exists() and SOFTMAX_LR_NPZ.exists()):
        return None
    vocab = load_json(SOFTMAX_VOCAB_JSON)
    idf = np.load(SOFTMAX_IDF_NPY)
    lrz = np.load(SOFTMAX_LR_NPZ, allow_pickle=True)
    coef = lrz["coef"]; intercept = lrz["intercept"]; classes = lrz["classes"]
    cfg = {"ngram_range": [1, 2], "lowercase": True, "strip_accents": "unicode"}
    if SOFTMAX_CFG_JSON.exists():
        cfg.update(load_json(SOFTMAX_CFG_JSON))
    tfidf = TfidfLite(
        vocab=vocab,
        idf=idf,
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        lowercase=bool(cfg.get("lowercase", True)),
        strip_accents=cfg.get("strip_accents", "unicode") == "unicode",
    )
    lr = SoftmaxLR(coef=coef, intercept=intercept, classes=list(classes))
    return {"tfidf": tfidf, "clf": lr, "classes": list(classes)}

def load_mlp_artifacts():
    needed = [MLP_PREPROC_JSON, MLP_TEXT_VOCAB_JSON, MLP_TEXT_IDF_NPY,
              MLP_SVD_NPZ, MLP_MAXABS_NPY, MLP_WEIGHTS_NPZ]
    if not all(p.exists() for p in needed):
        return None
    pre = load_json(MLP_PREPROC_JSON)
    text_vocab = load_json(MLP_TEXT_VOCAB_JSON)
    text_idf = np.load(MLP_TEXT_IDF_NPY)
    svdz = np.load(MLP_SVD_NPZ, allow_pickle=True)
    components = svdz["components"]
    maxabs = np.load(MLP_MAXABS_NPY)
    wz = np.load(MLP_WEIGHTS_NPZ, allow_pickle=True)
    W1, b1, W2, b2, W3, b3, classes = wz["W1"], wz["b1"], wz["W2"], wz["b2"], wz["W3"], wz["b3"], wz["classes"]
    cfg = pre.get("text_config", {"ngram_range": [1, 2], "lowercase": True, "strip_accents": "unicode"})

    tfidf = TfidfLite(
        vocab=text_vocab,
        idf=text_idf,
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        lowercase=bool(cfg.get("lowercase", True)),
        strip_accents=cfg.get("strip_accents", "unicode") == "unicode",
    )
    svd = TruncatedSVDLite(components=components)
    mlp = MLPInfer(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, classes=list(classes))

    return {
        "num_cols": pre.get("num_cols", []),
        "num_medians": pre.get("num_medians", []),
        "cat_cols": pre.get("cat_cols", []),
        "cat_categories": pre.get("cat_categories", {}),  # {col: [cat1, cat2, ...]}
        "text_cols": pre.get("text_cols", []),
        "tfidf": tfidf,
        "svd": svd,
        "maxabs": maxabs,
        "mlp": mlp,
        "classes": list(classes),
    }

def load_ensemble_weights():
    if ENSEMBLE_JSON.exists():
        d = load_json(ENSEMBLE_JSON)
        return float(d.get("w_softmax", 0.5)), float(d.get("w_mlp", 0.5))
    return 0.5, 0.5


# ------------------------------
# 主函数：predict_all
# ------------------------------
def predict_all(csv_path: str) -> List[str]:
    """
    读取 csv_path 并返回预测（字符串标签列表）。
    - 优先使用 Softmax 与 MLP 两分支融合；若某分支工件缺失，自动回退。
    """
    df = pd.read_csv(csv_path)

    # 载入工件
    smx = load_softmax_artifacts()
    mlp = load_mlp_artifacts()
    w_softmax, w_mlp = load_ensemble_weights()

    probs_list = []
    classes_ref = None

    # Softmax(TF-IDF + LR)
    if smx is not None:
        # 文本构造（逐行，无泄漏）
        texts = build_text_concat_for_softmax(df).astype(str).tolist()
        X_tfidf = smx["tfidf"].transform(texts)
        P = smx["clf"].predict_proba(X_tfidf)
        probs_list.append(("softmax", P, w_softmax, smx["classes"]))
        classes_ref = smx["classes"]

    # MLP(TF-IDF→SVD→MLP)
    if mlp is not None:
        # 与训练一致的列处理
        # 1) 多选展开（删除原长文本列）
        dfm = df.copy()
        dfm = expand_multi_select(dfm, MULTI_BEST,  prefix="best")
        dfm = expand_multi_select(dfm, MULTI_SUBOPT, prefix="subopt")

        # 2) 数值 / 类别 / 文本
        num_cols = mlp["num_cols"]
        X_num = impute_numeric(dfm, num_cols, mlp["num_medians"])

        cat_cols = mlp["cat_cols"]
        X_cat = ohe_categoricals(dfm, cat_cols, mlp["cat_categories"])

        text_cols = mlp["text_cols"]
        if text_cols:
            joined = dfm[text_cols].astype(str).fillna("").agg(" ".join, axis=1).tolist()
            X_txt = mlp["tfidf"].transform(joined)
            X_txt_svd = mlp["svd"].transform(X_txt)
        else:
            X_txt_svd = np.zeros((len(dfm), 0), dtype=np.float32)

        # 3) 拼接 + MaxAbs 缩放
        X_full = np.concatenate([X_num, X_cat, X_txt_svd], axis=1).astype(np.float32)
        X_full = maxabs_scale(X_full, mlp["maxabs"])

        # 4) 前向
        P = mlp["mlp"].predict_proba(X_full)
        probs_list.append(("mlp", P, w_mlp, mlp["classes"]))
        classes_ref = classes_ref or mlp["classes"]

    if not probs_list:
        # 未找到任何工件 → 返回全空预测（或抛错）。这里返回空字符串列表，避免评测器异常。
        return [""] * len(df)

    # 将各分支概率对齐到同一 classes 顺序（以第一分支为准）
    name0, P0, w0, classes0 = probs_list[0]
    classes_final = list(classes0)
    P_ens = np.zeros_like(P0, dtype=np.float32)
    # 累加第一个分支
    P_ens += w0 * P0

    # 合并其他分支
    for name, P, w, classes in probs_list[1:]:
        # 构造映射：把该分支的类别顺序对齐到 classes_final
        col_map = {c: i for i, c in enumerate(classes)}
        P_aligned = np.zeros_like(P0, dtype=np.float32)
        for j, c in enumerate(classes_final):
            if c in col_map:
                P_aligned[:, j] = P[:, col_map[c]]
            else:
                P_aligned[:, j] = 0.0
        P_ens += w * P_aligned

    # 归一化
    row_sum = P_ens.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    P_ens = P_ens / row_sum

    pred_idx = P_ens.argmax(axis=1)
    preds = [classes_final[i] for i in pred_idx]
    return preds


# ------------------------------
# 可选：命令行调用（评测器通常只 import 并调用 predict_all）
# ------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pred.py <path_to_csv>")
    else:
        out = predict_all(sys.argv[1])
        for p in out:
            print(p)
