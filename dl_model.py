# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score

from resnet import resnet18
from loggers import TrainLogger

logging.setLoggerClass(TrainLogger)


class DLModel(nn.Module):

    def __init__(self, n_embedding=256):
        super().__init__()
        
        self.n_embedding = n_embedding
        self.encoder = resnet18(n_embedding=n_embedding, in_channels=1)
        self.classifier = nn.Sequential(nn.Linear(n_embedding, n_embedding//2),
                                       nn.ReLU(),
                                       nn.Linear(n_embedding//2, 1))

        self.logger = logging.getLogger("dlmodel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")
        self = self.to(self.device)

    def loss_fn(self, predictions, targets, label):
        if label in ["diagnosis", "sex"]:
            # FIXME : pos_weight ?
            return F.binary_cross_entropy_with_logits(input=predictions.squeeze(), target=targets)
        else:
            return F.l1_loss(input=predictions.squeeze(), target=targets)

    def forward(self, x):
        z = self.encoder(x) 
        pred = self.classifier(z)
        return pred
    
    def get_embeddings(self, x):
        z = self.encoder(x)
        return z
    
    def configure_optimizers(self, **kwargs):
        return optim.Adam(self.parameters(), **kwargs)
    
    def fit(self, train_loader, nb_epochs, label,
            preproc, fold, chkpt_dir, **kwargs_optimizer):
        
        self.optimizer = self.configure_optimizers(**kwargs_optimizer)
        self.lr_scheduler = None
        self.save_hyperparameters(chkpt_dir)
        self.logger.reset_history()
        for epoch in range(nb_epochs):
            self.logger.info(f"Epoch: {epoch}")
            train_loss = 0

            self.train()
            self.logger.step()
            for batch in tqdm(train_loader, desc="train"):
                inputs = batch[preproc].to(self.device)
                targets = batch[label].to(self.device)
                loss = self.training_step(inputs, targets, label)
                train_loss += loss
            self.logger.reduce(reduce_fx="sum")
            self.logger.store(fold=fold, epoch=epoch, set="train", loss=train_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Train loss: {train_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'dlmodel_preproc-{preproc}_fold-{fold}_ep-{epoch}.pth'))
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(os.path.join(chkpt_dir,
                                     f'dlmodel_preproc-{preproc}_fold-{fold}_ep-{epoch}.pth'))
        self.logger.info(f"End of training: {self.logger.get_duration()}")

    def training_step(self, inputs, targets, label):
        self.optimizer.zero_grad()
        predictions = self(inputs)
        loss = self.loss_fn(predictions, targets, label)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()

    def test(self, loader, epoch, preproc, label, fold, chkpt_dir, save_y_pred=False):
        self.load_chkpt(os.path.join(chkpt_dir,
                                f'dlmodel_preproc-{preproc}_fold-{fold}_ep-{epoch}.pth'))
        test_loss = 0
        self.logger.reset_history()
        self.logger.step()
        y_pred, y_true = [], []
        for batch in tqdm(loader, desc="test"):
            inputs = batch[preproc].to(self.device)
            targets = batch[label].to(self.device)
            with torch.no_grad():
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets, label)
            test_loss += loss.item()
            y_pred.extend(outputs.squeeze().cpu().numpy())
            y_true.extend(targets.cpu().numpy())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        self.logger.info(f"Test loss : {test_loss:.2g}")
        self.logger.reduce(reduce_fx="sum")
        self.logger.store(fold=fold, epoch=epoch, set="test", loss=test_loss)

        r2 = r2_score(y_pred=y_pred, y_true=y_true)
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
        rmse = mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)
        self.logger.store(r2=r2, mean_absolute_error=mae, root_mean_squarred_error=rmse)
        """
        metrics = {"r2": r2_score,
                   "rmse": lambda y_pred, y_true: mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False),
                   "mae": mean_absolute_error}
        for name, metric in metrics.items():
            value = metric(y_pred=y_pred, y_true=y_true)
            self.logger.store(**{name: value})
                                                                                                                                                                    """
        self.logger.save(chkpt_dir, filename="_test")
        if save_y_pred:
            np.save(os.path.join(chkpt_dir, f"y_pred_fold-{fold}_epoch-{epoch}_test.npy"), y_pred)
            np.save(os.path.join(chkpt_dir, f"y_true_fold-{fold}_epoch-{epoch}_test.npy"), y_true)
        
    def save_hyperparameters(self, chkpt_dir):
        hp = {"n_embedding": self.n_embedding}
        with open(os.path.join(chkpt_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)

    def save_chkpt(self, filename):
        torch.save({"encoder": self.encoder.state_dict(),
                    "classifier": self.classifier.state_dict()},
                   filename)
    
    def load_chkpt(self, filename):
        chkpt = torch.load(filename, weights_only=True)
        status = self.encoder.load_state_dict(chkpt["encoder"], strict=False)
        self.logger.info(f"Loading encoder : {status}")
        status = self.classifier.load_state_dict(chkpt["classifier"], strict=False)
        self.logger.info(f"Loading classifier : {status}")
    