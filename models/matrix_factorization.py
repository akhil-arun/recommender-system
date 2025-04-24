import torch
import torch.nn as nn


class BiasedMF(nn.Module):
    def __init__(self, num_users, num_movies, num_age, num_gender, num_occ, num_genres, latent_dim=20):
        super(BiasedMF, self).__init__()

        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.movie_emb = nn.Embedding(num_movies, latent_dim)
        self.age_emb = nn.Embedding(num_age, latent_dim)
        self.gender_emb = nn.Embedding(num_gender, latent_dim)
        self.occ_emb = nn.Embedding(num_occ, latent_dim)
        self.genre_emb = nn.Embedding(num_genres, latent_dim)

        # Bias
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize weights: Model immediately overfits
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)
        nn.init.xavier_uniform_(self.age_emb.weight)
        nn.init.xavier_uniform_(self.gender_emb.weight)
        nn.init.xavier_uniform_(self.genre_emb.weight)
        nn.init.xavier_uniform_(self.occ_emb.weight)

    def forward(self, user_idx, movie_idx, occ_idx, age_idx, gender_idx, genre_vec):
        """Compute predicted ratings for a batch of user-item pairs."""

        genre_latent = genre_vec @ self.genre_emb.weight

        user_latent = self.user_emb(user_idx) + self.age_emb(age_idx) \
            + self.gender_emb(gender_idx) + self.occ_emb(occ_idx)

        item_latent = self.movie_emb(movie_idx) + genre_latent

        interaction = (user_latent * item_latent).sum(dim=1)

        # Final prediction
        return self.global_bias + self.user_bias(user_idx).squeeze(1) \
            + self.movie_bias(movie_idx).squeeze(1) + interaction


class MatrixFactorization(nn.Module):
    """Matrix Factorization model for collaborative filtering.

    This model learns user and item embeddings to predict user-item interactions.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        """Initialize the MatrixFactorization model.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_idx, item_idx):
        """Forward pass for the model.

        Args:
            user_idx (torch.Tensor): Tensor of user indices.
            item_idx (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: Predicted interaction scores for the given user-item pairs.
        """
        u = self.user_embeddings(user_idx)
        v = self.item_embeddings(item_idx)
        return (u * v).sum(dim=1)


class DeepMF(nn.Module):
    """Deep Matrix Factorization with an MLP-based interaction model."""

    def __init__(self, num_users, num_items, emb_dim):
        """Initialize the DeepMF model.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            emb_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, emb_dim)
        self.item_embeddings = nn.Embedding(num_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, user_ids, item_ids):
        """Forward pass for the model.

        Args:
            user_ids (torch.Tensor): Tensor of user indices.
            item_ids (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: Predicted interaction scores.
        """
        u = self.user_embeddings(user_ids)
        v = self.item_embeddings(item_ids)
        x = torch.cat([u, v], dim=-1)
        return self.mlp(x).squeeze()


class FeatureAwareDeepMF(nn.Module):
    """DeepMF model with user/item features, using float year and age."""

    def __init__(self, num_users, num_items, num_genders, num_occs, num_genres, emb_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, emb_dim)
        self.item_embeddings = nn.Embedding(num_items, emb_dim)

        self.gender_emb = nn.Embedding(num_genders, 4)
        self.occ_emb = nn.Embedding(num_occs, emb_dim)
        self.genre_emb = nn.Embedding(num_genres, emb_dim)  # multi-hot projection

        # Inputs: 5 vectors (emb_dim) + 4 (gender) + 2 floats (age, year)
        mlp_input_dim = (4 * emb_dim) + 4 + 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, user_ids, item_ids, gender_ids, occ_ids,
                year_vals, age_vals, genre_vecs):
        u = self.user_embeddings(user_ids)
        i = self.item_embeddings(item_ids)
        gender = self.gender_emb(gender_ids)
        occ = self.occ_emb(occ_ids)
        genre_proj = genre_vecs @ self.genre_emb.weight  # [B, emb_dim]

        year_vals = year_vals.unsqueeze(1)  # [B, 1]
        age_vals = age_vals.unsqueeze(1)    # [B, 1]

        x = torch.cat([u, i, occ, gender, genre_proj, year_vals, age_vals], dim=-1)
        return self.mlp(x).squeeze()
