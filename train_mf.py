import numpy as np
import torch
from tqdm import tqdm

from models.matrix_factorization import FeatureAwareDeepMF


class MFTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, epochs, device):
        """"Initialize the MFTrainer"""
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
    
    def criterion(self, pos_scores, neg_scores):
        """BPR loss function"""
        
        return -(pos_scores - neg_scores).sigmoid().log().mean()


    def train(self):
        """Trains the model for one epoch"""
        
        self.model.train()
        train_loss = 0
        for batch in self.train_loader:
            if isinstance(self.model, FeatureAwareDeepMF):
                user, pos, neg, gender, occ, year_pos, year_neg, age, _, genre_pos, genre_neg = batch
            
                user, pos, neg = user.to(self.device), pos.to(self.device), neg.to(self.device)
                gender, occ = gender.to(self.device), occ.to(self.device)
                year_pos, year_neg = year_pos.to(self.device), year_neg.to(self.device)
                age = age.to(self.device)
                genre_pos, genre_neg = genre_pos.to(self.device), genre_neg.to(self.device)

                pos_scores = self.model(user, pos, gender, occ, year_pos, age, genre_pos)
                neg_scores = self.model(user, neg, gender, occ, year_neg, age, genre_neg)
            
            else:
                user, pos, neg = batch
                user, pos, neg = user.to(self.device), pos.to(self.device), neg.to(self.device)
                pos_scores = self.model(user, pos)
                neg_scores = self.model(user, neg)
            
            loss = self.criterion(pos_scores, neg_scores)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        
        return train_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluates the model on the validation set"""
        
        self.model.eval()
        val_loss = 0
        for batch in self.val_loader:
            if isinstance(self.model, FeatureAwareDeepMF):
                user, pos, neg, gender, occ, year_pos, year_neg, age, _, genre_pos, genre_neg = batch
            
                user, pos, neg = user.to(self.device), pos.to(self.device), neg.to(self.device)
                gender, occ = gender.to(self.device), occ.to(self.device)
                year_pos, year_neg = year_pos.to(self.device), year_neg.to(self.device)
                age = age.to(self.device)
                genre_pos, genre_neg = genre_pos.to(self.device), genre_neg.to(self.device)

                pos_scores = self.model(user, pos, gender, occ, year_pos, age, genre_pos)
                neg_scores = self.model(user, neg, gender, occ, year_neg, age, genre_neg)
                
            else:
                user, pos, neg = batch
                user, pos, neg = user.to(self.device), pos.to(self.device), neg.to(self.device)
                pos_scores = self.model(user, pos)
                neg_scores = self.model(user, neg)

            loss = self.criterion(pos_scores, neg_scores)
            val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def fit(self):
        """Trains and evals the model for N epochs"""
        
        train_losses, val_losses = [], []
        for i in range(self.epochs):
            train_loss = self.train()
            train_losses.append(train_loss)
            
            val_loss = self.evaluate()
            val_losses.append(val_loss)
            
            print(f"Epoch {i+1}/{self.epochs} | Train Loss: {train_loss:.4f}  | Val Loss: {val_loss:.4f}")

        return train_losses, val_losses


