from pathlib import Path
import pandas as pd
import numpy as np
import random


# ─── 1. Loading ──────────────────────────────────────────────────────────


def load_movielens(data_dir):
    """
    Load MovieLens raw files into DataFrames.

    Args:
        data_dir (str): The directory containing the data files.

    Returns:
        tuple: DataFrames containing ratings, users, and movies.
    """
    data_dir = Path(data_dir)
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
        encoding="latin1"
    )
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        engine="python",
        encoding="latin1"
    )
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        names=["MovieID", "Title", "Genres"],
        engine="python",
        encoding="latin1"
    )
    return ratings, users, movies


# ─── 2. Cleaning & Filtering ────────────────────────────────────────────


def clean_and_filter(ratings, users, movies, rating_threshold=4):
    """
    Clean and filter the ratings DataFrame.

    Args:
        ratings (DataFrame): The ratings DataFrame.
        users (DataFrame): The users DataFrame.
        movies (DataFrame): The movies DataFrame.
        rating_threshold (int): The minimum rating threshold.

    Returns:
        tuple: Cleaned ratings, users, and movies DataFrames.
    """
    valid_ids = set(movies["MovieID"])
    ratings = ratings[ratings["MovieID"].isin(valid_ids)].copy()
    ratings = ratings[ratings["Rating"] >= rating_threshold].reset_index(
        drop=True
    )
    ratings["Datetime"] = pd.to_datetime(ratings["Timestamp"], unit="s")
    return ratings, users, movies


# ─── 3. Sequence Construction ───────────────────────────────────────────


def get_user_sequences(ratings):
    """
    Get user sequences from ratings.

    Args:
        ratings (DataFrame): The ratings DataFrame.

    Returns:
        dict: A dictionary mapping user_id to a list of movie_ids
        sorted by time
    """
    df = ratings.sort_values(["UserID", "Timestamp"])
    return df.groupby("UserID")["MovieID"].apply(list).to_dict()


# ─── 4. Train/Val/Test Split ────────────────────────────────────────────


def split_sequences(user_seqs, train_ratio=0.8, val_ratio=0.1):
    """
    Split user sequences into train, validation, and test sets.

    Args:
        user_seqs (dict): A dictionary mapping user_id to sequences.
        train_ratio (float): The ratio of the training set.
        val_ratio (float): The ratio of the validation set.

    Returns:
        dict: A dictionary mapping user_id to (train, val, test) sequences.
    """
    def _split(seq):
        n = len(seq)
        if n < 3:
            return None
        t1 = int(n * train_ratio)
        t2 = int(n * (train_ratio + val_ratio))
        return seq[:t1], seq[t1:t2], seq[t2:]

    splits = {}
    for u, seq in user_seqs.items():
        s = _split(seq)
        if s:
            splits[u] = s
    return splits


# ─── 5. Negative Sampling & Example Construction ────────────────────────


def sample_unseen(user_seq, global_movie_set, K=5):
    """Randomly sample K movies not in user_seq.

    Args:
        user_seq (list): The user's sequence of movie_ids.
        global_movie_set (set): The global set of movie_ids.
        K (int): The number of unseen movies to sample.

    Returns:
        list: A list of K unseen movie_ids.
    """
    unseen = list(global_movie_set - set(user_seq))
    return random.sample(unseen, K) if len(unseen) >= K else unseen


def build_examples(user_splits, global_movie_set, K=5, split="train"):
    """
    Build examples for the specified split.

    Args:
        user_splits (dict): A dictionary mapping user_id to (train, val, test)
        sequences.
        global_movie_set (set): The global set of movie_ids.
        K (int): The number of negative samples.
        split (str): The split to build examples for
                     ("train", "val", or "test").

    Returns:
        list: A list of examples containing prefix, positive, and negatives.
    """
    examples = []
    for u, (train_seq, val_seq, test_seq) in user_splits.items():
        # pick the right sequence
        seq = {"train": train_seq, "val": val_seq, "test": test_seq}[split]
        # choose starting index
        start = 2 if split == "train" else 1
        if len(seq) < start + 1:
            continue
        for t in range(start, len(seq)):
            prefix = seq[:t]
            examples.append({
                "UserID": u,
                "prefix": prefix,
                "positive": seq[t],
                "negatives": sample_unseen(prefix, global_movie_set, K)
            })
    return examples


# ─── 6. Padding & Masking ───────────────────────────────────────────────


def pad_sequences(examples, max_len=50, pad_val=0):
    """
    Pad sequences in-place for each example.

    Args:
        examples (list): A list of examples to pad.
        max_len (int): The maximum length of the padded sequence.
        pad_val (int): The value to use for padding.

    Returns:
        list: The list of examples with padded sequences.
    """
    for ex in examples:
        seq = ex["prefix"]
        L = min(len(seq), max_len)
        padded = np.full(max_len, pad_val, dtype=int)
        mask = np.zeros(max_len, dtype=int)
        padded[-L:] = seq[-L:]
        mask[-L:] = 1
        ex["padded_prefix"] = padded
        ex["mask"] = mask
    return examples


# ─── 7. Tabular Feature Tables ─────────────────────────────────────────


def build_user_table(users):
    """
    Build a user feature table.

    Args:
        users (DataFrame): The users DataFrame.

    Returns:
        DataFrame: The transformed user feature table.
    """
    age_map = {
        1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
        45: "45-49", 50: "50-55", 56: "56+"
    }
    df = users.copy()
    df["AgeBucket"] = df["Age"].map(age_map)
    df = df.drop("Zip-code", axis=1)
    return pd.get_dummies(df, columns=["Gender", "Occupation"],
                          prefix=["Gender", "Occ"])


def build_movie_table(movies):
    """
    Build a movie feature table.

    Args:
        movies (DataFrame): The movies DataFrame.

    Returns:
        DataFrame: The transformed movie feature table.
    """
    df = movies.copy()
    # multi-hot genres
    genre_dummies = df["Genres"].str.get_dummies(sep="|")
    df = df.drop("Genres", axis=1).join(genre_dummies)
    # extract year
    df[["Title", "Year"]] = df["Title"].str.extract(r'^(.*) \((\d{4})\)$')
    return df
