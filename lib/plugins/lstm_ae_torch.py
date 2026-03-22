from lib.plugin_base import PluginBase
from lib.model.lstm_auto_encoder import LSTMAutoencoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


classname = "LstmAeTorch"


class LstmAeTorch(PluginBase):
    def __init__(self):
        self.train_loss = 0
        pass

    def _pre_proc(self, X_train, X_test, y_train, y_test):
        pass

    def _post_proc(self, trained_model):
        pass

    def _str_(self):
        pass
    
    def _load_data(self, data_uri: str, params: dict = {}):
        return [], [], [], []
    
    def _model_save(self, model_name, trained_model, store_uri):
        pass
    
    def _metric_save(self, model_name, trained_model, store_uri):
        pass

    def _train(self, X_train, X_test=None, y_train=None, y_test=None, parameters={}):
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

        self.train_loss = 0.0

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
            self.train_loss = train_loss
            # if (epoch + 1) % 5 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(dataloader):.6f}")
    
        return model