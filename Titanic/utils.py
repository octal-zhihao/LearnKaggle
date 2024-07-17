import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(data):
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    return data

def scale_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)
