import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


class HousePricesDataset(Dataset):
    def __init__(self, data, train=True, augment=False):
        self.data = data
        self.train = train
        self.augment = augment
        self.features = self._preprocess(data)

    def _preprocess(self, data):
        # 填充数值型缺失值
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if data[col].isnull().any():
                if data[col].dtype == 'float64':
                    data[col].fillna(data[col].median(), inplace=True)
                elif data[col].dtype == 'int64':
                    data[col].fillna(data[col].median(), inplace=True)

        # 填充类别型缺失值
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if data[col].isnull().any():
                data[col].fillna(data[col].mode()[0], inplace=True)

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
        features = self.features[idx]
        if self.augment:
            features = self._augment(features)
        if self.train:
            label = self.data.iloc[idx]['SalePrice']
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(features, dtype=torch.float32)

    def _augment(self, features):
        noise = np.random.normal(0, 0.01, features.shape)
        features += noise
        return features