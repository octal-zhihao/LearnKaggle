import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

class HousePricesDataset(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train
        self.features = self._preprocess(data)

    def _preprocess(self, data):
        # 填充缺失值
        data['LotFrontage'].fillna(data['LotFrontage'].median(), inplace=True)
        data['MasVnrArea'].fillna(0, inplace=True)
        data['GarageYrBlt'].fillna(data['GarageYrBlt'].median(), inplace=True)

        if self.train:
            features = data.drop(['SalePrice'], axis=1).values
        else:
            features = data.values

        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.train:
            label = self.data.iloc[idx]['SalePrice']
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(self.features[idx], dtype=torch.float32)
