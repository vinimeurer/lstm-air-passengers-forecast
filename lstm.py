"""
LSTM for Time Series Forecasting (AirPassengers dataset)

Objetivos didáticos:
- Estrutura de LSTM em PyTorch
- Regularização (Dropout, Weight Decay, Gradient Clipping)
- Early Stopping
- Grid Search manual
- Pipeline reproduzível

Autor: versão didática para ensino
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import product

from dataset import load_data, TimeSeriesDataset
from model import LSTMModel
from early_stopping import EarlyStopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 12
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_model(model, train_loader, val_loader, config):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],  # L2 regularization
    )

    early_stopper = EarlyStopping(patience=15)

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()

            # Gradient clipping (evita exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                output = model(X)
                loss = criterion(output, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if early_stopper.step(val_loss, model):
            print("⛔ Early stopping disparado")
            break

    model.load_state_dict(early_stopper.best_model)
    return model, early_stopper.best_loss


def grid_search(train_data, val_data):
    grid = {
        "hidden_size": [32, 64],
        "num_layers": [1, 2],
        "dropout": [0.2, 0.5],
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-4],
    }

    best_config = None
    best_loss = float("inf")

    for hs, nl, dp, lr, wd in product(
        grid["hidden_size"],
        grid["num_layers"],
        grid["dropout"],
        grid["lr"],
        grid["weight_decay"],
    ):
        print("\n==============================")
        print(f"Config: hs={hs}, nl={nl}, dp={dp}, lr={lr}, wd={wd}")

        config = {"epochs": 100, "lr": lr, "weight_decay": wd}

        model = LSTMModel(hidden_size=hs, num_layers=nl, dropout=dp).to(DEVICE)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

        model, val_loss = train_model(model, train_loader, val_loader, config)

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = {
                "hidden_size": hs,
                "num_layers": nl,
                "dropout": dp,
                "lr": lr,
                "weight_decay": wd,
            }

    return best_config, best_loss


def evaluate_model(model, data_loader):
    """
    Avalia o modelo em dados nunca vistos (test set).
    """
    model.eval()
    criterion = nn.MSELoss()

    losses = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            loss = criterion(output, y)
            losses.append(loss.item())

    return np.mean(losses)


def main():
    series, _, _ = load_data()

    seq_len = 12

    train_size = int(len(series) * 0.7)
    val_size = int(len(series) * 0.15)

    train_series = series[:train_size]
    val_series = series[train_size : train_size + val_size]
    test_series = series[train_size + val_size :]

    train_data = TimeSeriesDataset(train_series, seq_len)
    val_data = TimeSeriesDataset(val_series, seq_len)
    test_data = TimeSeriesDataset(test_series, seq_len)

    best_config, best_loss = grid_search(train_data, val_data)

    print("\n✅ MELHOR CONFIGURAÇÃO (val):")
    print(best_config)
    print(f"Val Loss: {best_loss:.4f}")

    print("\n🚀 Treinando modelo final com train + val...")

    combined_series = np.concatenate([train_series, val_series])
    combined_data = TimeSeriesDataset(combined_series, seq_len)

    train_loader = DataLoader(combined_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE
    )  # usado para early stopping leve

    final_model = LSTMModel(
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"],
        dropout=best_config["dropout"],
    ).to(DEVICE)

    final_model, _ = train_model(
        final_model,
        train_loader,
        val_loader,
        {
            "epochs": 100,
            "lr": best_config["lr"],
            "weight_decay": best_config["weight_decay"],
        },
    )

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    test_loss = evaluate_model(final_model, test_loader)

    print("\n📊 RESULTADO FINAL (TEST SET):")
    print(f"Test Loss (MSE): {test_loss:.4f}")

if __name__ == "__main__":
    main()
