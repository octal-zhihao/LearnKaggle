from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.utils.data import Dataset

class HousePricesDataset(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train
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

        # 处理其他缺失值（如特殊值）
        # 你可以根据具体情况处理其他类型的缺失值，例如：
        # data['某列'].fillna('默认值', inplace=True)

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
