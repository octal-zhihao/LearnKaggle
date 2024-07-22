# author:octal
# time:2024/7/17
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from .dataset import TitanicDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.2, augment=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.augment = augment

    def setup(self, stage=None):
        train_data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

        all_data = pd.concat([train_data.drop(['Survived'], axis=1), test_data], ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(pd.get_dummies(all_data.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1),
                                  columns=['Sex', 'Embarked'], drop_first=True))

        self.train_dataset = TitanicDataset(train_data, scaler=scaler, train=True, augment=self.augment)
        self.test_dataset = TitanicDataset(test_data, scaler=scaler, train=False)

        train_size = int((1 - self.val_split) * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)