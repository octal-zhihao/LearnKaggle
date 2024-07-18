import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class TitanicDataset(Dataset):
    def __init__(self, data, scaler=None, train=True, augment=False):
        self.data = data
        self.train = train
        self.scaler = scaler
        self.augment = augment
        self.features = self._preprocess(data)

    def _preprocess(self, data):
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
        data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

        if self.train:
            features = data.drop(['Survived', 'PassengerId'], axis=1).values
            labels = data['Survived'].values
            if self.augment:
                features, labels = self._augment_data(features, labels)
        else:
            features = data.drop(['PassengerId'], axis=1).values

        if self.scaler:
            features = self.scaler.transform(features)

        if self.train:
            self.labels = labels

        return features

    def _augment_data(self, features, labels):
        # 在这里添加数据增强的方法，比如随机删除、增加噪声等
        # 这里是一个简单的例子，添加高斯噪声
        noise = np.random.normal(0, 0.01, features.shape)
        augmented_features = features + noise
        augmented_labels = labels

        # 将增强的数据和原始数据结合
        features = np.concatenate((features, augmented_features), axis=0)
        labels = np.concatenate((labels, augmented_labels), axis=0)

        return features, labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.train:
            label = self.labels[idx]
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(self.features[idx], dtype=torch.float32)
