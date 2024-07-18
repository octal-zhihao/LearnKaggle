# author:octal 
# time:2024/7/18


from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import torchmetrics

class MInterface(LightningModule):
    def __init__(self, input_dim, lr, num_heads, dropout_rate=0.5):
        super(MInterface, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads)

        self.r2_score = torchmetrics.R2Score()
        self.mean_absolute_error = torchmetrics.MeanAbsoluteError()
        self.mean_squared_error = torchmetrics.MeanSquaredError()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        # 注意力机制
        x = x.unsqueeze(1)  # (batch_size, seq_len, embed_dim)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)  # (batch_size, embed_dim)

        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        r2 = self.r2_score(y_hat, y.unsqueeze(1))
        mae = self.mean_absolute_error(y_hat, y.unsqueeze(1))
        mse = self.mean_squared_error(y_hat, y.unsqueeze(1))

        self.log('train_r2', r2)
        self.log('train_mae', mae)
        self.log('train_mse', mse)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))

        r2 = self.r2_score(y_hat, y.unsqueeze(1))
        mae = self.mean_absolute_error(y_hat, y.unsqueeze(1))
        mse = self.mean_squared_error(y_hat, y.unsqueeze(1))

        self.log('val_r2', r2)
        self.log('val_mae', mae)
        self.log('val_mse', mse)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
