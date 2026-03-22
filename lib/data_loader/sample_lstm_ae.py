import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def _data_loader_tensor():
    # 가상의 시계열 데이터 생성 (1000개 샘플, 10 시퀀스, 1 특성)
    data = np.sin(np.linspace(0, 100, 1000)).reshape(-1, 1)
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
    X = np.array(X)

    # PyTorch 텐서로 변환 및 GPU 이동
    X_tensor = torch.FloatTensor(X).to(device)
    dataset = TensorDataset(X_tensor, X_tensor) # 입력과 타겟이 같음
    return dataset, None, dataset, None


def _data_loader_temp_gen():
    # 0~100 사이의 사인 파형 생성
    t = np.linspace(0, 100, 1000)
    data = np.sin(t) + np.random.normal(0, 0.1, 1000)
    data = data.reshape(-1, 1)

    # 데이터 스케일링 (0~1 사이로 정규화 또는 표준화)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 시계열 데이터의 구조를 변환 (Windowing)
    # 10개 시점(Time Step)을 입력받아 10개 시점을 예측하도록 구성
    def create_sequences(data, time_steps=10):
        xs = []
        for i in range(len(data) - time_steps):
            x = data[i:(i + time_steps)]
            xs.append(x)
        return np.array(xs)

    TIME_STEPS = 10
    X = create_sequences(data_scaled, TIME_STEPS)
    return X, None, X, None


def _data_load():
    return _data_loader_temp_gen()
