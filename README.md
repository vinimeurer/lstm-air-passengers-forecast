# LSTM Time Series Forecasting (PyTorch)

Implementação didática de uma LSTM para previsão de séries temporais usando PyTorch, baseada no dataset clássico AirPassengers.

## Objetivo

Demonstrar um pipeline completo de treino de redes recorrentes com boas práticas de Deep Learning:

- Pré-processamento de séries temporais
- Modelagem com LSTM
- Regularização
- Early Stopping
- Ajuste de hiperparâmetros (Grid Search)

## Arquitetura

- LSTM multi-layer
- Dropout interno e externo
- Camada fully connected para regressão

## Técnicas aplicadas

- Normalização dos dados
- Dropout
- Weight Decay (regularização L2)
- Gradient Clipping (evita exploding gradients)
- Early Stopping
- Grid Search manual

## Dataset

AirPassengers:
- Série temporal mensal de passageiros aéreos
- Problema clássico de forecasting
- **Fonte**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
