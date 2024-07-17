# author:octal 
# time:2024/7/17
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from .model import TitanicModel

class MInterface(pl.LightningModule):
    def __init__(self, input_dim, lr=0.003):
        super(MInterface, self).__init__()
        self.model = TitanicModel(input_dim)
        self.lr = lr
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

