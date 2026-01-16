# Sentiment / Survey Response Classification (Ensemble + Exportable Artifacts)

这个 repo 的目标是把一个“课程/个人项目”包装成 **招聘官一眼能复现、能评估、能部署推理** 的形态：

- **训练**：`Softmax(TF‑IDF + LogisticRegression)` + `MLP(Structured + TF‑IDF→SVD)`  
- **集成**：两路概率加权平均（默认 0.5 / 0.5）  
- **导出**：生成 `pred.py` 可直接加载的 artifacts（纯 numpy 推理，无 sklearn/torch 依赖）  
- **报告**：自动生成 **confusion matrix** 和 **per-class F1 bar chart**（加分项）  
- **baseline**：提供一个简单 KNN baseline 用于对照（加分项）

> 你可以把这个 repo 直接放 GitHub：它的亮点不是“又一个分类模型”，而是 **end-to-end 的可复现 + 可导出推理 + 指标/可视化**。

---

## Repo 结构（必做 1）

```
.
├── src/
│   ├── train.py               # 训练 + 评估 + 导出 + 生成报告（主入口）
│   ├── export_artifacts.py    # 工件导出逻辑（供 pred.py 使用）
│   ├── features.py            # 特征构造（与 pred.py 对齐，避免 training/inference drift）
│   ├── reporting.py           # 指标 + confusion matrix + per-class F1 图
│   └── config.py              # 列名/超参集中管理
├── baselines/
│   └── knn_baseline.py        # baseline 对照（加分 3）
├── pred.py                    # 纯 numpy 推理入口（读取 artifacts/）
├── artifacts/                 # 训练后生成的工件（不提交 Git）
├── reports/                   # 训练后生成的图与指标（不提交 Git）
└── data/
    └── training_data_clean.csv  # 你自己的数据（不提交 Git）
```

---

## Quickstart

### 1) 安装依赖
```bash
python -m venv .venv
source .venv/bin/activate   # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 放数据
把你的 CSV 放到：
- `data/training_data_clean.csv`

CSV 至少需要：
- `label`（目标类别）
- `student_id`（可选；用于 group-aware split，避免同一个人泄漏到 test）

### 3) 一键训练 + 导出 + 生成报告（必做 2-3 + 加分 2）
```bash
python -m src.train --data data/training_data_clean.csv --seed 42
```

你会得到：
- `artifacts/` 下的一系列文件（供 `pred.py` 推理使用）
- `reports/metrics.json`
- `reports/confusion_matrix.png`
- `reports/per_class_f1.png`

### 4) 推理（加载 artifacts）
```bash
python pred.py path/to/unlabeled.csv > preds.txt
```

> `pred.py` 暴露了一个函数：`predict_all(csv_path) -> List[str]`

---

## Baseline 对照（加分 3）
```bash
python -m baselines.knn_baseline --data data/training_data_clean.csv --seed 42
```

建议你在 README 里维护一个结果表（训练一次就能填）：

| Model | Accuracy | Macro-F1 |
|------|----------|----------|
| KNN baseline | (run script) | (run script) |
| Softmax (TF-IDF + LR) | (from reports/metrics.json) | (from reports/metrics.json) |
| MLP (Structured + TF-IDF→SVD) | (from reports/metrics.json) | (from reports/metrics.json) |
| **Ensemble** | **(from reports/metrics.json)** | **(from reports/metrics.json)** |

---

## Artifacts 说明（导出文件名固定）
这些文件名是为了让 `pred.py` 可以直接读取：

Softmax 分支：
- `softmax_vocab.json`
- `softmax_idf.npy`
- `softmax_lr.npz`
- `softmax_config.json`

MLP 分支：
- `mlp_preproc.json`
- `mlp_text_vocab.json`
- `mlp_text_idf.npy`
- `mlp_svd.npz`
- `mlp_maxabs.npy`
- `mlp_weights.npz`

Ensemble 权重：
- `ensemble.json`

---

## 常见问题
- **我没有 GPU 可以跑吗？**  
  可以。默认会自动选择 `cuda`（如果可用），否则走 CPU。

- **为什么要导出 artifacts？**  
  这能把项目从“只能在 notebook 里跑”升级为“可部署推理服务的雏形”，对 MLE/AI Engineer 简历非常加分。

