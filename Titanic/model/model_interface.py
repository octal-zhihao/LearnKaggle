# author:octal 
# time:2024/7/17

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import torchmetrics

class MInterface(LightningModule):
    def __init__(self, input_dim, lr, num_heads):
        super(MInterface, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.lr = lr
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads)

        self.f1_score = torchmetrics.F1Score(task='binary', threshold=0.5)
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.precision = torchmetrics.Precision(task='binary')
        self.recall = torchmetrics.Recall(task='binary')

    def forward(self, x):
        x = self.relu(self.layer_1(x))

        # 注意力机制
        x = x.unsqueeze(1)  # (batch_size, seq_len, embed_dim)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)  # (batch_size, embed_dim)

        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.layer_3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        preds = torch.round(y_hat)
        f1 = self.f1_score(preds, y.unsqueeze(1))
        acc = self.accuracy(preds, y.unsqueeze(1))
        precision = self.precision(preds, y.unsqueeze(1))
        recall = self.recall(preds, y.unsqueeze(1))

        self.log('train_acc', acc)
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        preds = torch.round(y_hat)
        f1 = self.f1_score(preds, y.unsqueeze(1))
        acc = self.accuracy(preds, y.unsqueeze(1))
        precision = self.precision(preds, y.unsqueeze(1))
        recall = self.recall(preds, y.unsqueeze(1))

        self.log('val_acc', acc)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
