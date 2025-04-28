import torch.nn as nn
import torch


class SASR(nn.Module):
    def __init__(self, num_users, num_items, max_length=50, d_model=64, n_head=2, num_layers=2, dropout=0.2,
                 device='mps'):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, d_model)
        self.item_embeddings = nn.Embedding(num_items, d_model, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_length = max_length
        self.device = device

    def forward(self, users, prefix):
        # embedding_users = self.user_embeddings(users)
        position_indices = torch.arange(prefix.shape[-1]).unsqueeze(0).expand(prefix.shape[0], -1).to(self.device)
        item_embeddings = self.item_embeddings(prefix) + self.positional_embeddings(position_indices)
        embeddings = item_embeddings
        mask = nn.Transformer.generate_square_subsequent_mask(self.max_length).to(self.device)

        encoded = self.encoder(embeddings, mask=mask)
        return encoded

    def get_scores(self, encoded_seq_next, candidates, user):
        context = encoded_seq_next
        candidate_embeddings = self.item_embeddings(candidates)
        scores = torch.sum(context * candidate_embeddings, -1)
        return scores
