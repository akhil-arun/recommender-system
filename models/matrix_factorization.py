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
        """
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)
        nn.init.xavier_uniform_(self.age_emb.weight)
        nn.init.xavier_uniform_(self.gender_emb.weight)
        nn.init.xavier_uniform_(self.genre_emb.weight)
        nn.init.xavier_uniform_(self.occ_emb.weight)
        """

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
