from models.sequential_dcn_v2 import DCNV2_Sequential
from models.vanilla_nn import TwoLayerNet
from trainer import Trainer
from data_utils.datasets import CustomDataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import yaml
from config import Config
import pandas as pd

parser = argparse.ArgumentParser(description="trainer")
parser.add_argument('--config_file', type=str, default='configs/config_nn.yaml', help="path to YAML config")
parser.add_argument('--output_dir', type=str, default=None,
                    help="path to output directory (optional); defaults to outputs/model_name")
args = parser.parse_args()

with open(args.config_file, 'r') as file:
    config_dict = yaml.safe_load(file)
    config = Config(config_dict=config_dict)
df = pd.read_csv('data/dataset.csv')
target_column = 'rating'

if config.network.model == 'nn':
    # Generate sparse input.
    X_sparse_input = {}

    # Generate dense input.
    X_dense_columns = list(set(df.columns)- {target_column})
    X_dense_input = torch.tensor(df[X_dense_columns].values, dtype=torch.float32)
    y = torch.tensor(df[target_column].values, dtype=torch.float32)
elif config.network.model == 'dcn_v2_sequential':
    # Generate sparse input.
    sparse_feature_info = {
        # name: (vocab_size, embed_size)
        "uid": (10000, 64),       # 10,000 users, 64-dim embedding
        "movie_id": (5000, 64),        # 5,000 items, 64-dim embedding
        # "age_sparse": (7, 8),        # 5,000 items, 64-dim embedding
    }
    sparse_columns = sparse_feature_info.keys()
    X_sparse_input = {
        name: torch.tensor(df[name])
        for name, (vocab_size, embed_size) in sparse_feature_info.items()
    }

    # Generate dense input.
    dense_columns = list(set(df.columns) - set(sparse_columns) - {target_column})
    num_dense_features = len(dense_columns)
    X_dense_input = torch.tensor(df[dense_columns].values)
    y = torch.tensor(df[target_column].values, dtype=torch.float32)

dataset = CustomDataset(X_sparse_input, X_dense_input, y)
loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)

if config.network.model == 'nn':
    model = TwoLayerNet(input_dim=len(X_dense_columns), hidden_size=784, num_classes=1)
elif config.network.model == 'dcn_parallel':
    model = TwoLayerNet(input_dim=len(X_dense_columns), hidden_size=784, num_classes=1)
elif config.network.model == 'dcn_v2_sequential':
    model = DCNV2_Sequential(sparse_feature_info=sparse_feature_info, num_dense_features=num_dense_features)

trainer = Trainer(model, None, config, loader, float(config.train.lr))

trainer.fit()

