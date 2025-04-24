import random
import numpy as np
import torch
import pandas as pd


# ─── Metric Functions ───────────────────────────────────────────


def hit_at_k(rank: int, k: int = 10) -> float:
    """Calculate Hit@k metric.

    Args:
        rank (int): The rank of the positive item.
        k (int, optional): The cutoff rank. Defaults to 10.

    Returns:
        float: 1.0 if rank is less than or equal to k, else 0.0.
    """
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int = 10) -> float:
    """Calculate NDCG@k metric.

    Args:
        rank (int): The rank of the positive item.
        k (int, optional): The cutoff rank. Defaults to 10.

    Returns:
        float: The NDCG score.
    """
    return 1.0 / np.log2(rank + 1) if rank <= k else 0.0


def mrr(rank: int) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        rank (int): The rank of the positive item.

    Returns:
        float: The MRR score.
    """
    return 1.0 / rank


def average_precision(rank: int) -> float:
    """Calculate Average Precision (AP).

    Args:
        rank (int): The rank of the positive item.

    Returns:
        float: The Average Precision score.
    """
    return 1.0 / rank


# ─── Default Negative Sampler ────────────────────────────────────


def uniform_negative_sampler(prefix: list,
                             global_items: set,
                             num_samples: int) -> list:
    """Sample negative items uniformly.

    Args:
        prefix (list): List of seen items.
        global_items (set): Set of all item IDs.
        num_samples (int): Number of negative samples to draw.

    Returns:
        list: List of sampled negative item IDs.
    """
    unseen = list(global_items - set(prefix))
    if len(unseen) < num_samples:
        return unseen
    return random.sample(unseen, num_samples)


# ─── Core Evaluation Function ────────────────────────────────────


def evaluate_ranking_model(
    model: torch.nn.Module,
    user_splits: dict,
    global_items: set,
    device: torch.device,
    *,
    candidate_size: int = 100,
    k: int = 10,
    negative_sampler=uniform_negative_sampler
) -> dict:
    """Evaluate a ranking model.

    Args:
        model (torch.nn.Module): The trained recommender model.
        user_splits (dict): Dictionary mapping users to (train_seq, val_seq,
                            test_seq).
        global_items (set): Full set of all item IDs.
        device (torch.device): Device for inference.
        candidate_size (int, optional): Total candidates = 1 pos +
                                         (candidate_size-1) negs. Defaults to 100.
        k (int, optional): Cutoff for Hit@k and NDCG@k. Defaults to 10.
        negative_sampler (function, optional): Function to sample negative IDs.
                                               Defaults to uniform_negative_sampler.

    Returns:
        dict: Dictionary with averaged metrics: Hit@k, NDCG@k, MRR, MAP.
    """
    model.eval()
    hits, ndcgs, mrrs, aps = [], [], [], []

    for user, (train_seq, val_seq, test_seq) in user_splits.items():
        if not test_seq:
            continue
        # 1) Build the “prefix” and the held‑out positive item
        prefix = train_seq + val_seq
        pos_item = test_seq[0]

        # 2) Sample negatives
        negs = negative_sampler(prefix, global_items - {pos_item},
                                candidate_size - 1)

        # 3) Build candidate list
        candidates = [pos_item] + negs

        # 4) Score all candidates in one forward pass
        users_t = torch.tensor([user] * candidate_size, dtype=torch.long,
                               device=device)
        items_t = torch.tensor(candidates, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model(users_t, items_t).cpu().numpy()
        # 5) Compute the rank of the positive item (index 0 before sorting)
        ranking = np.argsort(-scores)
        rank = np.where(ranking == 0)[0][0] + 1  # 1‑based

        # 6) Accumulate metrics
        hits.append(hit_at_k(rank, k))
        ndcgs.append(ndcg_at_k(rank, k))
        mrrs.append(mrr(rank))
        aps.append(average_precision(rank))

    # 7) Return average over all users
    return {
        f"Hit@{k}": np.mean(hits),
        f"NDCG@{k}": np.mean(ndcgs),
        "MRR": np.mean(mrrs),
        "MAP": np.mean(aps)
    }




def evaluate_DCNV2Model(
    model: torch.nn.Module,
    user_splits: dict,
    global_items: set,
    device: torch.device,
    *,
    candidate_size: int = 100,
    k: int = 10,
    negative_sampler=uniform_negative_sampler,
    df: pd.DataFrame
) -> dict:
    """Evaluate a ranking model.

    Args:
        model (torch.nn.Module): The trained recommender model.
        user_splits (dict): Dictionary mapping users to (train_seq, val_seq,
                            test_seq).
        global_items (set): Full set of all item IDs.
        device (torch.device): Device for inference.
        candidate_size (int, optional): Total candidates = 1 pos +
                                         (candidate_size-1) negs. Defaults to 100.
        k (int, optional): Cutoff for Hit@k and NDCG@k. Defaults to 10.
        negative_sampler (function, optional): Function to sample negative IDs.
                                               Defaults to uniform_negative_sampler.

    Returns:
        dict: Dictionary with averaged metrics: Hit@k, NDCG@k, MRR, MAP.
    """
    model.eval()
    hits, ndcgs, mrrs, aps = [], [], [], []

    temp=0
    for user, (train_seq, val_seq, test_seq) in user_splits.items():
        if not test_seq:
            continue

        # 1) Build the “prefix” and the held‑out positive item
        prefix = train_seq + val_seq
        pos_item = test_seq[0]

        # 2) Sample negatives
        global_items = set(df['movie_id'].unique())
        negs = negative_sampler(prefix, global_items - {pos_item},
                                candidate_size - 1)

        # 3) Build candidate list
        candidates = [pos_item] + negs

        # 4) Score all candidates in one forward pass
        users_t = torch.tensor([user] * candidate_size, dtype=torch.long,
                               device=device)
        items_t = torch.tensor(candidates, dtype=torch.long, device=device)
        with torch.no_grad():
            #####################################################################################
            # Main logic. 
            df = df[(df['movie_id'].isin(items_t.tolist()))]
            df = df.drop_duplicates(subset='movie_id', keep='first')

            df.loc[:,'uid']=user

            df.head()

            # Sort on movie id in items_t
            order = {v: i for i, v in enumerate(items_t)}
            df['__key'] = df['movie_id'].map(order)
            df = df.sort_values('__key').drop(columns='__key')

            sparse_feature_info = {
                # name: (vocab_size, embed_size)
                "uid": (10000, 64),       # 10,000 users, 64-dim embedding
                "movie_id": (5000, 64),        # 5,000 items, 64-dim embedding
                # "age_sparse": (7, 8),        # 5,000 items, 64-dim embedding
            }
            sparse_columns = sparse_feature_info.keys()
            X_sparse_input = {
                name: torch.tensor(df[name].values)
                for name, (a, b) in sparse_feature_info.items()
            }
            target_column = 'rating'

            # Generate dense input.
            dense_columns = list(set(df.columns) - set(sparse_columns) - {target_column})

            X_sparse_input = X_sparse_input
            X_dense_input = torch.tensor(df[dense_columns].values)

            # end of main logic.
            #####################################################################

            # scores = model(users_t, items_t).cpu().numpy()
            if temp==0:
                # print("X_sparse_input: ", X_sparse_input.shape)
                # print("X_dense_input: ", X_dense_input.shape)
                print("X_sparse_input: ", X_sparse_input)
                print("dense_columns: ", X_dense_input)

            scores = model(X_sparse_input, X_dense_input).cpu().numpy()

            if temp==0:
                print("scores: ", scores.shape)
                print("scores: ", scores)
            temp=temp+1
        # 5) Compute the rank of the positive item (index 0 before sorting)
        ranking = np.argsort(-scores)
        rank = np.where(ranking == 0)[0][0] + 1  # 1‑based

        # 6) Accumulate metrics
        hits.append(hit_at_k(rank, k))
        ndcgs.append(ndcg_at_k(rank, k))
        mrrs.append(mrr(rank))
        aps.append(average_precision(rank))

    # 7) Return average over all users
    return {
        f"Hit@{k}": np.mean(hits),
        f"NDCG@{k}": np.mean(ndcgs),
        "MRR": np.mean(mrrs),
        "MAP": np.mean(aps)
    }


def evaluate_featureaware_model(
    model: torch.nn.Module,
    user_splits: dict,
    global_items: set,
    user_feat: dict,
    movie_genre_vec: dict,
    movie_year: dict,
    device: torch.device,
    *,
    candidate_size: int = 100,
    k: int = 10,
    negative_sampler=uniform_negative_sampler
) -> dict:
    """Evaluate FeatureAwareDeepMF: pulls occ, age, gender, year and genre for each candidate."""
    model.eval()
    hits, ndcgs, mrrs, aps = [], [], [], []
    
    for user, (train_seq, val_seq, test_seq) in user_splits.items():
        if not test_seq:
            continue

        # prefix & positive
        prefix = train_seq + val_seq
        pos_item = test_seq[0]

        # negatives
        negs = negative_sampler(prefix, global_items - {pos_item}, candidate_size - 1)
        candidates = [pos_item] + negs

        # build feature lists
        occ_id, age_val, gender_id = user_feat[user]
        user_ids   = [user] * candidate_size
        gender_ids = [gender_id] * candidate_size
        occ_ids    = [occ_id] * candidate_size
        age_vals   = [age_val] * candidate_size
        year_vals  = [movie_year[mid] for mid in candidates]
        genre_vecs = [movie_genre_vec[mid] for mid in candidates]

        # to tensors
        users_t   = torch.tensor(user_ids,   dtype=torch.long,   device=device)
        items_t   = torch.tensor(candidates, dtype=torch.long,   device=device)
        genders_t = torch.tensor(gender_ids, dtype=torch.long,   device=device)
        occs_t    = torch.tensor(occ_ids,    dtype=torch.long,   device=device)
        years_t   = torch.tensor(year_vals,  dtype=torch.float,  device=device)
        ages_t    = torch.tensor(age_vals,   dtype=torch.float,  device=device)
        genres_t  = torch.stack([torch.tensor(v, dtype=torch.float) for v in genre_vecs], dim=0).to(device)

        # forward
        with torch.no_grad():
            scores = model(
                users_t, items_t,
                genders_t, occs_t,
                years_t, ages_t,
                genres_t
            ).cpu().numpy()

        # compute ranking
        ranking = np.argsort(-scores)
        rank = np.where(ranking == 0)[0][0] + 1  # 1-based

        # accumulate
        hits.append(hit_at_k(rank, k))
        ndcgs.append(ndcg_at_k(rank, k))
        mrrs.append(mrr(rank) )
        aps.append(average_precision(rank))

    return {
        f"Hit@{k}": np.mean(hits),
        f"NDCG@{k}": np.mean(ndcgs),
        "MRR": np.mean(mrrs),
        "MAP": np.mean(aps)
    }