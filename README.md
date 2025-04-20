# MovieLens 1M Sequence‑Aware Recommender Benchmark

## 1. Project Objective

Benchmark three generations of recommenders on a sequence next‑item task:

1. **Biased Matrix Factorization** (classical baseline)  
2. **Tabular Deep Models** (DeepFM and/or DCNv2)  
3. **Sequence‑Aware Transformers** (SASRec and/or PinnerFormer‑style encoder)  

**Goal**: Quantify the lift each stage provides on top‑N ranking quality, calibrated across identical data splits and negative‑sampling schemes.

---

## 2. Dataset & Problem Setup

**Dataset**: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)  
- **Ratings**: 1,000,209  
- **Users**: 6,040  
- **Movies**: 3,900  

**Signals**:  
- Explicit star ratings (1–5)  
- Timestamps  
- User demographics  
- Movie genres  

**Task**:  
Sequence next‑item prediction — given a user’s chronological prefix, predict the next movie they rate ≥ 4 stars.

- **Positives**: Held‑out last ≥4‑star interaction per user  
- **Negatives**: 99 random unseen movies per test case (uniform sampling)

---

## 3. Data Preparation Pipeline

1. Load and clean `ratings`, `users`, `movies`. Drop invalid MovieIDs.
2. Filter to ratings ≥ 4 (positive signal).
3. Sort each user’s ratings chronologically → sequences `s = [m₁, ..., m_T]`.
4. Split per user:
   - Train = first 80%  
   - Validation = next 10%  
   - Test = final 10%
5. Construct training sequences:
   - For each prefix `s[:t]` (t ≥ 2):  
     - Positive = `s[t]`  
     - K negatives = `sample_unseen(user, K)` (default K = 5)
6. Pad and mask sequences for transformer batches (max length L = 50).
7. Build feature tables for tabular models:
   - User: age bucket, occupation  
   - Movie: multi‑hot genres

---

## 4. Model Stack & Training Details

| Tier             | Core Idea                  | Loss         | Init             | Notes                                           |
|------------------|----------------------------|--------------|------------------|-------------------------------------------------|
| **MF**           | Biased latent factors      | BPR (1+K)    | Xavier           | Fast baseline; used to warm-start deep models   |
| **DeepFM/DCNv2** | Sparse-dense cross layers  | BCE          | Random           | Ingests demographics + genres                   |
| **SASRec**       | Transformer encoder        | CE over 1+K  | Pre-trained MF   | Self-attention (L=50, 2 layers)                 |
| **PinnerFormer-lite** | SASRec + conv-mixing      | CE           | SASRec weights    | (stretch goal) Adds local context with depthwise convolutions  |

---

## 5. Evaluation Protocol

1. For each user’s test prefix, score the 100-item candidate set.
2. Compute and average user-level metrics:
   - **Primary**: Hit@10, NDCG@10  
   - **Secondary**: MRR, MAP  
3. [stretch] Analyze cold‑start (<10 train ratings) vs power users (≥50).
4. [stretch] Statistical significance:
   - Bootstrap 1,000× NDCG@10  
   - Report 95% CI and paired t‑test

---

## 6. Planned Ablation Studies

| Axis                  | Variation                          | Hypothesis                                                   |
|------------------------|------------------------------------|---------------------------------------------------------------|
| Negative sample K      | 1, 5, 20, 50                       | Higher K → sharper rank signal, but memory-intensive          |
| Sequence length L      | 20, 50, 100                        | Longer histories benefit transformers more than MF            |
| Positive threshold     | Rating ≥5 vs ≥4                   | Stricter positive reduces data, may lower recall              |
| Feature sets           | +demographics, +genres, both, none | DCNv2 expected to gain most from feature richness             |
| Embedding warm-start   | MF → deep vs random               | Warm-start helps convergence                                 |
| Model depth            | SASRec: 2 vs 4 layers              | Little gain beyond 2 layers for small dataset                 |

---

# Code structure and Usage

This repository provides end-to-end code for preprocessing the MovieLens dataset, training models, and evaluating their performance. It’s structured to let you plug in new models easily, while reusing existing preprocessing and evaluation pipelines.

## 1. Environment Setup

1. Install Conda (if you don’t have it).
2. Create environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate recommender-env
   ```
3. Ensure Python 3.10+ is active.

---

## 2. Data Preparation

1. Download and place the MovieLens raw files (`movies.dat`, `ratings.dat`, `users.dat`) into the `data/` folder.
2. Verify file integrity (MD5 or file sizes) if needed.

---

## 3. Preprocessing (`data_utils/preprocess.py`)

The `preprocess.py` script handles:
- Loads raw MovieLens .dat files into Pandas DataFrames
- Filters out low-rating interactions (e.g., only keep ratings ≥ 4)
- Sorts and splits each user’s interaction history into train/val/test
- Returns clean dictionaries and sequences per user
- Performs negative sampling during evaluation set construction

### Usage (within Python):
```python
from preprocess import load_movielens, clean_and_filter, get_user_sequences, split_sequences, build_datasets

ratings, users, movies = load_movielens("data/")
ratings, users, movies = clean_and_filter(ratings, users, movies, rating_threshold=4)

user_seqs = get_user_sequences(ratings)
splits = split_sequences(user_seqs, train_ratio=0.8, val_ratio=0.1)

all_items = set(movies.MovieID)
train_data, val_data, test_data = build_datasets(splits, all_items, candidate_size=100)
```

Refer to the function docstring for parameter details.

---

## 4. Model Definitions (`models/id_only_mf.py`)

Contains the matrix factorization baseline:
- `MatrixFactorization`: simple embedding-based MF with dot-product (no features)

### To add a new model:
1. Create a new file in `models/`, e.g. `my_model.py`.
2. Define a class with `.forward(user_ids, item_ids)` or `.forward(batch)`.
3. Ensure it accepts the same input format as other models.
4. Import and instantiate it in your experiment script or notebook.

---

## 5. Evaluation Pipeline (`evaluation.py`)

This module provides reusable evaluation tools for recommender systems using implicit feedback. It supports Hit@K, NDCG@K, MRR, and MAP, and evaluates models using a parameterized negative sampling routine. 	
- Standard ranking metrics: Hit@K, NDCG@K, MRR, MAP
- Compatible with any PyTorch model: accepts model(user_tensor, item_tensor) interface
- Supports custom negative samplers. By default it samples negatives from items the user hasn’t seen
- Designed for sequence-aware recommendation, but works with general recommendation models

### Signature:
```python
evaluate_ranking_model(
    model,                  # PyTorch model
    user_splits,            # dict[user_id] = (train_seq, val_seq, test_seq)
    global_items,           # set of all item IDs
    device,                 # torch.device('cuda') or 'cpu'
    *,
    candidate_size=100,     # 1 positive + N-1 negatives
    k=10,                   # Metric cutoff
    negative_sampler=...    # Function to generate negatives
) -> dict
```

### Usage
```python
from evaluation import evaluate_ranking_model
from torch import device

metrics = evaluate_ranking_model(
    model=my_model,
    user_splits=my_user_split_dict,
    global_items=set_of_all_items,
    device=torch.device("cuda"),
    candidate_size=100,
    k=10
)

print(metrics)
```

---

## 6. Running Experiments (`experiments/baseline.ipynb`)

The notebook shows:
1. Loading and preprocessing data
2. Instantiating the baseline MF model
3. Training loop with BCE loss
4. Evaluating metrics on validation/test
5. Plotting learning curves

I recommend you create new notebooks (e.g. preprended with your initials) to run experiments for the report.

---

## 8. Adding and Testing New Models

1. Implement your model in `models/your_model.py`.
2. In a new notebook or script:
   - Import `preprocess_movielens`, `SequentialDataset`, `evaluate_model`.
   - Create data loaders.
   - Instantiate your model, optimizer, and loss.
   - Train with a loop similar to the baseline notebook.
   - Call `evaluate_model` after each epoch.

This structure lets you focus on model innovation without rewriting data or eval code.

---

## 9. Contributing

- Follow PEP8 style in new modules.
- Write google style docstrings for new functions.
- Add unit tests under a `tests/` folder (future).

---