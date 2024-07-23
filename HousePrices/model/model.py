# author:octal 
# time:2024/7/22
import torch.nn as nn


class HousePriceModel(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(HousePriceModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 16)
        self.layer5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)  # 在第一层后添加dropout
        x = self.relu(self.layer2(x))
        x = self.dropout(x)  # 在第二层后添加dropout
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.layer5(x)
        return x

