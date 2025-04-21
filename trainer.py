import torch
import torch.nn.functional as F
from tqdm import tqdm
from data_utils.data_utils import set_seed


class Trainer:
    def __init__(self, model, optimizer, cfg, train_loader, lr, val_loader=None):
        self.model = model.to(cfg.train.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr) if not optimizer else optimizer
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        set_seed(seed=42)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for sparse, dense, target in tqdm(self.train_loader):
            # sparse = sparse.to(self.cfg.train.device)
            dense = dense.to(self.cfg.train.device)
            target = target.to(self.cfg.train.device)

            pred = self.model(sparse, dense)
            loss = F.binary_cross_entropy(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for sparse, dense, target in tqdm(self.train_loader):
                # sparse = sparse.to(self.cfg.train.device)
                dense = dense.to(self.cfg.train.device)
                target = target.to(self.cfg.train.device)

                pred = self.model(sparse, dense)
                loss = F.binary_cross_entropy(pred, target)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self):
        for epoch in range(self.cfg.train.num_epochs):
            train_loss = self.train_epoch()
            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}")

            if self.val_loader:
                val_loss = self.evaluate()
                print(f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}")
