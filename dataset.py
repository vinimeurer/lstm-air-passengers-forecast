import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
)


def load_data():
    df = pd.read_csv(URL)
    series = df["Passengers"].values.astype(np.float32)

    # Normalização (essencial para LSTM)
    mean = np.mean(series)
    std = np.std(series)
    series = (series - mean) / std

    return series, mean, std


class TimeSeriesDataset(Dataset):
    """
    Transforma série temporal em pares (X, y):
    X: sequência passada
    y: próximo valor
    """

    def __init__(self, series, seq_len):
        self.X = []
        self.y = []

        for i in range(len(series) - seq_len):
            self.X.append(series[i : i + seq_len])
            self.y.append(series[i + seq_len])

        self.X = torch.tensor(self.X).unsqueeze(-1)  # (N, seq_len, 1)
        self.y = torch.tensor(self.y).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
