from models.vanilla_nn import TwoLayerNet
from trainer import Trainer
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
sparse_columns = list(set(df.columns) - {'rating'})
## Do a train test split here
X_sparse = torch.tensor(df[sparse_columns].values, dtype=torch.float32)
y = torch.tensor(df[target_column].values, dtype=torch.float32)

dataset = TensorDataset(X_sparse, y)
loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)

if config.network.model == 'nn':
    model = TwoLayerNet(input_dim=len(sparse_columns), hidden_size=784, num_classes=1)
elif config.network.model == 'dcn_parallel':
    model = TwoLayerNet(input_dim=len(sparse_columns), hidden_size=784, num_classes=1)

trainer = Trainer(model, None, config, loader, float(config.train.lr))

trainer.fit()