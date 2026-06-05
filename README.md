# Compact FashionMNIST Retrieval and Compression

Course: Artificial Neural Network  
Student: Alperen Davran  
Lecturers: Arian Sabaghi and José Antonio Oramas Mogrovejo  
University of Antwerp

This repository contains my final FashionMNIST project:

1. Part A compares StandardCNN and SeparableCNN.
2. Part B trains a triplet-loss retrieval embedding model initialized from the selected Part A checkpoint.
3. Part C evaluates global L1 pruning and RKD distillation into CompactSeparableCNN.

The final report is `main.pdf`.

## Source Layout

Core project files follow the suggested staged structure from the brief:

| File | Stage | Purpose |
|---|---|---|
| `src/config.py` | shared | Fixed seed, model names, subset sizes, and batch-size defaults |
| `src/data.py` | shared | Shared FashionMNIST loaders for classification, retrieval training, validation, and evaluation |
| `src/models.py` | shared | `StandardCNN`, completed `SeparableCNN`, and fixed `CompactSeparableCNN` |
| `src/losses.py` | Part B/C | Contrastive, triplet, batch-hard triplet, and RKD losses |
| `src/train_classifier.py` | Part A | StandardCNN vs SeparableCNN classifier comparison |
| `src/train_metric.py` | Part B | Retrieval embedding training initialized from the selected Part A checkpoint |
| `src/prune_and_distill.py` | Part C | Global L1 pruning and RKD distillation of the retrieval model |
| `src/visualize_retrieval.py` | Part B | Query plus nearest-neighbor retrieval examples |
| `src/visualize_embeddings.py` | Part B | t-SNE, per-class retrieval diagnostics, and retrieval confusion matrix |
| `src/run_project.py` | all | Runs the required A -> B -> C pipeline |

Two small diagnostic scripts are included because their outputs are discussed in
the report: `src/pruning_study.py` for the pruning-method comparison and
`src/gradcam.py` for classifier Grad-CAM examples. They do not change the
required A -> B -> C dependency chain.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_project.py
```

Local runtime folders such as `.venv/`, `data/`, and `__pycache__/` are not part
of the submission.
