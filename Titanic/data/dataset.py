import pandas as pd
from torch.utils.data import Dataset
import torch

class TitanicDataset(Dataset):
    def __init__(self, data, scaler=None, train=True):
        self.data = data
        self.train = train
        self.scaler = scaler
        self.features = self._preprocess(data)

    def _preprocess(self, data):
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
        data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

        if self.train:
            features = data.drop(['Survived', 'PassengerId'], axis=1).values
        else:
            features = data.drop(['PassengerId'], axis=1).values

        if self.scaler:
            features = self.scaler.transform(features)

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.train:
            label = self.data.iloc[idx]['Survived']
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(self.features[idx], dtype=torch.float32)
