import os
import json
import requests
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Dropout
from torch.utils.data import DataLoader, Dataset

# Гиперпараметры
input_size = 1
hidden_size = 400
num_layers = 4
output_size = 1
num_epochs = 500
learning_rate = 0.0001
sequence_length = 10
batch_size = 256
dropout_p = 0.3
period = 70


def fetch_data():
    url = 'https://api.eternity-business.ru/web/v1/anal/analyzed'
    headers = {
        'Authorization': 'Bearer NhrWATjSXlckMTFz'
    }
    params = {
        'page': 1,
        'limit': 500
    }

    data = []

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        pages = int(response.json()['_meta']['pageCount'])
        data += response.json()['items']

        for i in range(2, pages + 1):
            params['page'] = i
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data += response.json()['items']
        return data
    else:
        print(f'Request failed with status code {response.status_code}')
        return None


def load_data(file_path='cached_data.json'):
    if (os.path.exists(file_path)):
        print("Loading data from cache...")
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        print("Fetching data from API...")
        data = fetch_data()
        if data:
            with open(file_path, 'w') as f:
                json.dump(data, f)
    return data


def generate_data(seq_length=120):
    time = np.arange(seq_length)
    data = np.sin(time / 5) + np.random.normal(0, 0.1, seq_length)
    return data


class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (self.data[index:index + self.seq_length], self.data[index + self.seq_length])


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


class StockPricePredictor:
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate, num_epochs, sequence_length,
                 batch_size, dropout_prob):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Используется GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("Используется CPU")
        self.model = LSTM(input_size, hidden_size, num_layers, output_size, dropout_prob).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def prepare_data(self, data):
        df = pd.DataFrame(data)
        df = df[df['city'] == 'Москва']
        df = df.drop(
            columns=['city', 'id', 'bdAddTime', 'ad_squares', 'anal_type', 'data_range', 'data_date', 'data_table',
                     'max_price_ad', 'min_price_ad', 'average_chronological_ad', 'average_arithmetic_ad',
                     'average_medial_ad',
                     'medial_ad', 'upper_medial_ad', 'lower_medial_ad', 'price_sum_ad', ])
        df = df.astype('float32')

        df = df[['medial_m', 'min_price_m', 'max_price_m', 'average_chronological_m', 'average_arithmetic_m',
                 'average_medial_m', 'upper_medial_m', 'lower_medial_m', 'popular_square', 'price_sum_m', 'ad_quantity',
                 'inPlace']]

        original_data = df['medial_m'].values
        print(original_data)
        scaled_data = self.scaler.fit_transform(original_data.reshape(-1, 1)).reshape(-1)
        train_data = scaled_data[:period]
        test_data = scaled_data[period:]

        train_dataset = StockDataset(train_data, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = StockDataset(test_data, self.sequence_length)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, test_loader, original_data

    def train_model(self, train_loader):
        self.model.train()
        for epoch in range(self.num_epochs):
            for inputs, targets in train_loader:
                inputs = inputs.view(-1, self.sequence_length, self.input_size).to(self.device).float()
                targets = targets.view(-1, 1).to(self.device).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def predict_next_two_days(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            # Извлекаем последнюю последовательность данных из тестового набора
            inputs, _ = next(iter(test_loader))
            inputs = inputs.view(-1, self.sequence_length, self.input_size).to(self.device).float()
            # Получаем предсказание на следующие два дня
            two_days_prediction = self.model(inputs).squeeze().cpu().numpy()
        # Возвращаем предсказание на следующие два дня
        return self.scaler.inverse_transform(two_days_prediction.reshape(-1, 1)).reshape(-1)

    def test_model(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.view(-1, self.sequence_length, self.input_size).to(self.device).float()
                targets = targets.view(-1, 1).to(self.device).float()

                outputs = self.model(inputs)
                predictions.append(outputs.item())
                actuals.append(targets.item())

        actuals = self.scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).reshape(-1)
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).reshape(-1)

        return actuals, predictions

    def plot_results(self, original_data, actuals, predictions):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(original_data)), original_data, label='Данные для тренировки', alpha=0.5)
        plt.plot(np.arange(len(original_data) - len(actuals), len(original_data)), actuals, label='Реальная медиальня')
        plt.plot(np.arange(len(original_data) - len(predictions), len(original_data)), predictions,
                 label='Предсказанная медиальня')
        plt.legend()
        plt.xlabel('Дни')
        plt.ylabel('Медиальная за кв.м')
        plt.show()

    def run(self, data):
        train_loader, test_loader, original_data = self.prepare_data(data)
        self.train_model(train_loader)
        actuals, predictions = self.test_model(test_loader)
        next_two_days = self.predict_next_two_days(test_loader)
        print(next_two_days)
        self.plot_results(original_data, actuals, predictions)
        print(f'Актуальные данные: {original_data}')
        print(f'Предсказания {predictions}')
        print(f'Предсказано дней: {len(predictions)}')
        mpe = np.mean(
            np.abs((original_data[-len(predictions):] - predictions) / original_data[-len(predictions):])) * 100
        print(f'Средний процент ошибок: {mpe:.2f}%')
                # Предсказание на два дня вперед без данных
        print("Предсказание на два дня вперед:")
        next_two_days = self.predict_next_two_days_without_data()
        print(next_two_days)

    def predict_next_two_days_without_data(self):
        self.model.eval()
        with torch.no_grad():
            # Создаем пустой тензор для начала предсказания
            inputs = torch.zeros(1, self.sequence_length, self.input_size).to(self.device).float()
            # Предсказываем каждый следующий день и обновляем входные данные
            predictions = []
            for _ in range(2):
                output = self.model(inputs)
                # Добавляем предсказание в список
                predictions.append(output.item())
                # Обновляем входные данные, удаляя первый элемент и добавляя предсказание в конец
                inputs = torch.cat((inputs[:, 1:, :], output.unsqueeze(0)), 1)  # Removed unnecessary unsqueeze
        # Инвертируем предсказанные значения и возвращаем их
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).reshape(-1)

if __name__ == '__main__':
    data = load_data()
    predictor = StockPricePredictor(input_size, hidden_size, num_layers, output_size, learning_rate, num_epochs,
                                    sequence_length, batch_size, dropout_p)
    predictor.run(data)
