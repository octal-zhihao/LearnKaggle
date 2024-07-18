# author:octal 
# time:2024/7/18
# dataset.py
import pandas as pd
from torch.utils.data import Dataset
import torch

class HousePricesDataset(Dataset):
    def __init__(self, data, scaler=None, train=True):
        self.data = data
        self.train = train
        self.scaler = scaler
        self.features = self._preprocess(data)

    def _preprocess(self, data):
        # 填充缺失值
        data['LotFrontage'].fillna(data['LotFrontage'].median(), inplace=True)
        data['MasVnrArea'].fillna(0, inplace=True)
        data['GarageYrBlt'].fillna(data['GarageYrBlt'].median(), inplace=True)

        # 填充类别特征的缺失值
        for col in data.select_dtypes(include=['object']).columns:
            data[col].fillna(data[col].mode()[0], inplace=True)

        # 删除不必要的列
        data.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

        # 独热编码
        data = pd.get_dummies(data, drop_first=True)

        if self.train:
            features = data.drop(['SalePrice'], axis=1).values
        else:
            features = data.values

        if self.scaler:
            features = self.scaler.transform(features)

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.train:
            label = self.data.iloc[idx]['SalePrice']
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(self.features[idx], dtype=torch.float32)
