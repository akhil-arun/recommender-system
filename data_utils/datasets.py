from torch.utils.data import Dataset
import torch
from collections import defaultdict


class MFTrainDataset(Dataset):
    """Dataset for training with user-item interactions.

    Attributes:
        examples (list): A list of dicts, each containing user ID, positive
            item, and sampled negative items.
        num_negatives (int): The number of negative samples to return.
    """

    def __init__(self, examples, num_negatives=1):
        """Initializes the dataset with examples and number of negatives.

        Args:
            examples (list): A list of dicts, each having:
                - "UserID": int
                - "positive": int (the next item)
                - "negatives": list[int] (sampled negatives)
            num_negatives (int): How many negatives to return (we’ll just pick
                the first).
        """
        self.examples = examples
        self.num_negatives = num_negatives

    def __len__(self):
        """Returns the total number of training triples.

        Returns:
            int: Total number of (user, positive, negative) training triples.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """Fetches the training sample at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the user ID, positive item, and first
                negative item as tensors.
        """
        ex = self.examples[idx]
        user = ex["UserID"]
        pos = ex["positive"]
        neg = ex["negatives"][:self.num_negatives]  # e.g. [j₁, j₂, …]
        padded_prefix = ex["padded_prefix"]

        # return as tensors for PyTorch
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(padded_prefix, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            # just take the first negative
            torch.tensor(neg[0], dtype=torch.long)
        )


class FeatureAwareDeepMFDataset(Dataset):
    """Dataset for FeatureAwareDeepMF with float age/year + genre + embeddings."""

    def __init__(self, examples, user_feat, movie_genre_vec, movie_year):
        self.examples = examples
        self.user_feat = user_feat
        self.movie_genre_vec = movie_genre_vec
        self.movie_year = movie_year

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        u = ex["UserID"]
        pos = ex["positive"]
        neg = ex["negatives"][0]

        occ_id, age_val, gender_id = self.user_feat[u]
        genre_pos = self.movie_genre_vec[pos]
        genre_neg = self.movie_genre_vec[neg]
        year_pos = self.movie_year[pos]
        year_neg = self.movie_year[neg]

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
            torch.tensor(gender_id, dtype=torch.long),
            torch.tensor(occ_id, dtype=torch.long),
            torch.tensor(year_pos, dtype=torch.float),
            torch.tensor(year_neg, dtype=torch.float),
            torch.tensor(age_val, dtype=torch.float),
            torch.tensor(age_val, dtype=torch.float),  # same age for pos/neg
            torch.tensor(genre_pos, dtype=torch.float),
            torch.tensor(genre_neg, dtype=torch.float),
        )



class DCNV2Dataset(Dataset):
    def __init__(self, sparse_input: dict, dense_input: torch.Tensor, labels: torch.Tensor):
        self.sparse_input = sparse_input
        self.dense_input = dense_input
        self.labels = labels
        self.keys = list(sparse_input.keys())
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sparse = {key: self.sparse_input[key][idx] for key in self.keys}
        dense = self.dense_input[idx]
        label = self.labels[idx]
        return sparse, dense, label


class SASRTrainDataset(MFTrainDataset):
    """Dataset for training with user-item interactions.

    Attributes:
        examples (list): A list of dicts, each containing user ID, positive
            item, and sampled negative items.
        num_negatives (int): The number of negative samples to return.
    """

    def __getitem__(self, idx):
        """Fetches the training sample at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the user ID, positive item, and first
                negative item as tensors.
        """
        ex = self.examples[idx]
        user = ex["UserID"]
        pos = ex["positive"]
        neg = ex["negatives"][:self.num_negatives]  # e.g. [j₁, j₂, …]
        padded_prefix = ex["padded_prefix"]
        # return as tensors for PyTorch
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            # just take the first negative
            torch.tensor(neg[0], dtype=torch.long),
            torch.tensor(padded_prefix, dtype=torch.long)
        )

