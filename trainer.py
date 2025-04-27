import torch
import torch.nn.functional as F
from tqdm import tqdm
from data_utils.data_utils import set_seed
from torchmetrics import AUROC, Accuracy, Precision, Recall
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer, cfg, train_loader, lr, val_loader=None):
        self.model = model.to(cfg.train.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr) if not optimizer else optimizer
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        set_seed(seed=42)
        self.device = self.cfg.train.device
        print("Trainer init device:", self.device)
        self.metric_auc       = AUROC(task="binary").to(self.device)
        self.metric_acc       = Accuracy(task="binary").to(self.device)
        self.metric_prec      = Precision(task="binary",average='macro', num_classes=2).to(self.device)
        self.metric_recall    = Recall(task="binary", average='macro', num_classes=2).to(self.device)
        self.train_losses=[]
        self.val_losses=[]


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for sparse, dense, target in tqdm(self.train_loader):
            # sparse = sparse.to(self.cfg.train.device)
            sparse = {k: v.to(self.cfg.train.device) for k, v in sparse.items()}
            dense = dense.to(self.cfg.train.device)
            target = target.to(self.cfg.train.device)

            pred = self.model(sparse, dense)
            loss = F.binary_cross_entropy(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        self.metric_auc.reset()
        self.metric_acc.reset()
        self.metric_prec.reset()
        self.metric_recall.reset()
        total_samples=0
        with torch.no_grad():
            for sparse, dense, target in tqdm(self.val_loader):
                # sparse = sparse.to(self.cfg.train.device)
                sparse = {k: v.to(self.cfg.train.device) for k, v in sparse.items()}
                dense = dense.to(self.cfg.train.device)
                target = target.to(self.cfg.train.device)

                pred = self.model(sparse, dense)
                loss = F.binary_cross_entropy(pred, target)
                total_loss += loss.item()

                # update metrics
                self.metric_auc.update(pred, target)
                self.metric_acc.update((pred >= 0.5), target)
                self.metric_prec.update((pred >= 0.5), target)
                self.metric_recall.update((pred >= 0.5), target)
        decimals=4
        avg_loss   = total_loss / len(self.val_loader)
        auc        = round(self.metric_auc.compute().item(), decimals)
        accuracy   = round(self.metric_acc.compute().item(), decimals)
        precision  = round(self.metric_prec.compute().item(), decimals)
        recall     = round(self.metric_recall.compute().item(), decimals)
        self.val_losses.append(avg_loss)
        return {
            'loss': avg_loss,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

    def fit(self):
        for epoch in range(self.cfg.train.num_epochs):
            train_loss = self.train_epoch()
            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}")

            if self.val_loader:
                print(self.evaluate())
        if self.val_loader:
            plt.plot(self.train_losses, label="Train Loss")
            plt.plot(self.val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.ylim(0, 1)
            plt.title("Training & Validation Loss for DCN-V2 Model")
            plt.legend()
            plt.grid(True)
            plt.show()
