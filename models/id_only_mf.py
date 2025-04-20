import torch.nn as nn


class MatrixFactorization(nn.Module):
    """Matrix Factorization model for collaborative filtering.

    This model learns user and item embeddings to predict user-item
    interactions.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        """Initialize the MatrixFactorization model.

        Args:
            num_users (int): The number of users.
            num_items (int): The number of items.
            embedding_dim (int): The dimension of the embedding vectors.
        """
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_idx, item_idx):
        """Forward pass for the model.

        Args:
            user_idx: Indices of the users.
            item_idx: Indices of the items.

        Returns:
            Tensor: Predicted interaction scores for the given user-item pairs.
        """
        u = self.user_embeddings(user_idx)
        v = self.item_embeddings(item_idx)
        return (u * v).sum(dim=1)
