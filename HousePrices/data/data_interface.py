import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from .dataset import HousePricesDataset


class DInterface(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        train_data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

        # 合并训练数据和测试数据进行独热编码，确保特征一致
        all_data = pd.concat([train_data.drop(['SalePrice'], axis=1), test_data], ignore_index=True)
        all_data = pd.get_dummies(all_data, drop_first=True)

        train_data_encoded = all_data.iloc[:train_data.shape[0], :]
        test_data_encoded = all_data.iloc[train_data.shape[0]:, :]

        scaler = StandardScaler()
        scaler.fit(train_data_encoded)

        # 指定后缀来避免列名冲突
        self.train_dataset = HousePricesDataset(train_data.join(train_data_encoded, rsuffix='_train'), scaler=scaler,
                                                train=True)
        self.test_dataset = HousePricesDataset(test_data.join(test_data_encoded, rsuffix='_test'), scaler=scaler,
                                               train=False)

        train_size = int((1 - self.val_split) * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
