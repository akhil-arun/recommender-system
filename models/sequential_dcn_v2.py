import torch.nn as nn
import torch


class CrossLayersV2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        xl_w = (xl @ self.weights) + self.bias
        return x0 * xl_w + xl


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayersV2(input_dim) for _ in range(num_layers)])

    def forward(self, x0):
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        return xl


class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = nn.ModuleList()
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)


class DCNV2_Sequential(nn.Module):
    def __init__(self, sparse_feature_info, num_dense_features, cross_layers=3, deep_hidden_dims=[256, 128]):
        super().__init__()
        self.sparse_embeddings = nn.ModuleDict()
        for name, (vocab_size, embed_dim) in sparse_feature_info.items():
            self.sparse_embeddings[name] = nn.Embedding(vocab_size, embed_dim)

        self.embedding_total_dim = num_dense_features + sum(
            embed_dim for _, (_, embed_dim) in sparse_feature_info.items())
        self.cross_layers = cross_layers
        self.cross_network = CrossNetwork(self.embedding_total_dim, self.cross_layers)
        self.deep_network = DeepNetwork(self.embedding_total_dim, deep_hidden_dims)
        self.final_linear_layer = nn.Sequential(
            nn.Linear(deep_hidden_dims[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, sparse_input, dense_input):
        embedding_list = []
        for name in sparse_input.keys():
            # shape: [batch, embed_dim]
            embedding_list.append(self.sparse_embeddings[name](sparse_input[name]))
        dense_embeddings = torch.cat([dense_input, torch.cat(embedding_list, -1)], -1)
        cl_output = self.cross_network(dense_embeddings)
        dl_output = self.deep_network(cl_output)
        output = self.final_linear_layer(dl_output)
        return output.squeeze()