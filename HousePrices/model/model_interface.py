from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import torchmetrics
from .model import HousePriceModel

class MInterface(LightningModule):
    def __init__(self, input_dim, lr, dropout_rate):
        super(MInterface, self).__init__()
        self.model = HousePriceModel(input_dim, dropout_rate)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.mean_squared_error = torchmetrics.MeanSquaredError()

    def forward(self, x):
        x = self.model(x)
        return x

    def calculate_log_rmse(self, y_hat, y):
        # Avoid log(0) by adding a small epsilon value
        epsilon = 1e-10
        y_hat_log = torch.log(y_hat + epsilon)
        y_log = torch.log(y + epsilon)
        rmse = torch.sqrt(self.mean_squared_error(y_hat_log, y_log))
        return rmse

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        # Calculate log RMSE
        log_rmse = self.calculate_log_rmse(y_hat.squeeze(), y)

        self.log('train_log_rmse', log_rmse, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        # Calculate log RMSE
        log_rmse = self.calculate_log_rmse(y_hat.squeeze(), y)

        self.log('val_log_rmse', log_rmse, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
