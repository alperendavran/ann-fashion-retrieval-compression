# Reproduce

GitHub repository:
https://github.com/alperendavran/ann-fashion-retrieval-compression

## Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Full Required Pipeline

```bash
python src/run_project.py
```

This runs:

1. `train_classifier.py`: Part A, StandardCNN vs SeparableCNN.
2. `train_metric.py`: Part B, triplet retrieval initialized from Part A.
3. `prune_and_distill.py`: Part C, 40% global L1 pruning and RKD distillation.
4. `visualize_retrieval.py` and `visualize_embeddings.py`: required retrieval examples, t-SNE, per-class diagnostics, and retrieval confusion matrix.

Report: `main.pdf` (LaTeX source: `main.tex`).

## Optional Diagnostic Ablation

The report also includes a contrastive-loss control run under the same Part B
protocol as the triplet teacher (same LR, batch size, and epochs; only the loss
differs). It is kept separate from the required A->B->C pipeline so that the
Part C teacher remains the triplet model.

```bash
python src/train_metric.py \
  --losses contrastive \
  --output-dir outputs/part_b_contrastive_ablation \
  --epochs 8 \
  --learning-rate 5e-5 \
  --batch-size 256 \
  --no-metric-augmentation
```

The report also includes a pruning-method study (local/global x L1/random at
40% sparsity, plus a global L1 sparsity sweep) on the Part B teacher. It is a
diagnostic and does not change the selected compressed model.

```bash
python src/pruning_study.py
```

The report also includes Grad-CAM heatmaps for the selected Part A classifier,
used to explain which regions drive the upper-garment confusions. It is a
diagnostic and does not modify any trained model.

```bash
python src/gradcam.py
```

Important outputs:

- `outputs/part_a/`
- `outputs/part_b/`
- `outputs/part_b_contrastive_ablation/`
- `outputs/part_c/` (includes `pruning_study.json`)
- `outputs/figures/retrieval/query_0.png`
- `outputs/figures/retrieval/query_17.png`
- `outputs/figures/retrieval/query_42.png`
- `outputs/figures/embedding_tsne.png`
- `outputs/figures/retrieval_confusion_matrix.png`
- `outputs/figures/gradcam.png`

## Packaging Note

Do not submit `.venv/`, `data/`, `__pycache__/`, or smoke-test outputs. They are
local runtime artifacts, not project source or required results.
