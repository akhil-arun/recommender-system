from torch.utils.data import Dataset
import torch


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

        # return as tensors for PyTorch
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos,  dtype=torch.long),
            # just take the first negative
            torch.tensor(neg[0], dtype=torch.long)
        )


class BiasedMFDataset(Dataset):
    def __init__(self, examples, user_feat, movie_genre_vec):
        self.exs = examples
        self.uf = user_feat
        self.mg = movie_genre_vec

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        u = ex["UserID"]
        pos = ex["positive"]
        neg = ex["negatives"][0]

        occ, age, gender = self.uf[u]
        genre_pos = self.mg[pos]
        genre_neg = self.mg[neg]

        return (
            torch.tensor(u,      dtype=torch.long),
            torch.tensor(pos,    dtype=torch.long),
            torch.tensor(neg,    dtype=torch.long),
            torch.tensor(occ,    dtype=torch.long),
            torch.tensor(age,    dtype=torch.long),
            torch.tensor(gender, dtype=torch.long),
            torch.tensor(genre_pos, dtype=torch.float),
            torch.tensor(genre_neg, dtype=torch.float),
        )
