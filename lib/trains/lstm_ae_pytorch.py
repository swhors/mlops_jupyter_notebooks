import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from lib.model.lstm_auto_encoder import LSTMAutoencoder


def _train(X_train, y_train=None, parameters={}):
    epochs=parameters.get('epochs', 20)
    batch_size=parameters.get('batch_size', 32)
    validation_split=parameters.get('validation_split', 0.1)
    shuffle=parameters.get('shuffle', False)

    # 2. 데이터셋 준비
    dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=shuffle)

    # 3. 모델, 손실 함수, 옵티마이저 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(seq_len=50, n_features=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    g_train_loss = 0.0

    # 4. 모델 학습
    model.train()
    print(f"학습 시작 ({device})")
    for epoch in range(epochs):
        train_loss = 0.0
        for bacth_x, _ in dataloader:
            bacth_x = bacth_x.to(device)

            # forward
            output = model(bacth_x)
            loss = criterion(output, bacth_x)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
        g_train_loss = train_loss
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(dataloader):.6f}")
    
    results = {
        "model": model,
        "train_loss": g_train_loss,
        "accuracy": None, # Autoencoder는 일반적으로 정확도 대신 손실을 평가 지표로 사용
    }
    return results
