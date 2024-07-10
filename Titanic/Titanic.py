#!/usr/bin/env python
# coding: utf-8

# 导入库和数据：
#
# 从Kaggle下载Titanic数据集，并导入到我们的Python环境中。

# In[50]:


import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# 自定义Dataset类

# In[52]:


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


# 数据预处理和划分：

# In[53]:


# 合并训练和测试数据以确保相同的预处理
all_data = pd.concat([train_data.drop(['Survived'], axis=1), test_data], ignore_index=True)

# 标准化
scaler = StandardScaler()
scaler.fit(pd.get_dummies(all_data.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1), columns=['Sex', 'Embarked'], drop_first=True))

# 创建训练和验证数据集
train_dataset = TitanicDataset(train_data, scaler=scaler, train=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 划分训练集和验证集
train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
val_loader = DataLoader(train_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(val_indices))


# 定义模型：
#
# 使用PyTorch Lightning定义模型。

# In[54]:


class TitanicModel(pl.LightningModule):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.layer_1 = nn.Linear(8, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.layer_3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        return optimizer


# 训练模型：
#
# 使用PyTorch Lightning的Trainer进行训练。

# In[55]:


early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=15)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode='min',
    filename='TitanicModel-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    verbose=True
)

# 配置CSVLogger
csv_logger = CSVLogger(save_dir='logs/', name='titanic')

model = TitanicModel()

trainer = Trainer(
    max_epochs=50,
    check_val_every_n_epoch=1,
    logger=csv_logger,
    callbacks=[early_stopping_callback, checkpoint_callback]
)

trainer.fit(model, train_loader, val_loader)

# 加载最优模型
best_model_path = checkpoint_callback.best_model_path
best_model = TitanicModel.load_from_checkpoint(best_model_path)


# 对测试数据进行预测：
#
# 同样的预处理步骤，然后用训练好的模型进行预测。

# In[56]:


# 创建测试数据集
test_dataset = TitanicDataset(test_data, scaler=scaler, train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测
best_model.eval()
test_pred = []
with torch.no_grad():
    for batch in test_loader:
        preds = best_model(batch)
        preds = (preds.numpy() > 0.5).astype(int)
        test_pred.extend(preds)

# 保存预测结果
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': np.array(test_pred).ravel()})
submission.to_csv('submission.csv', index=False)



