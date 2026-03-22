## Import Dependencies
import os
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def _train(X_train, y_train=None, parameters={}):
    
    epochs=parameters.get('epochs', 20)
    batch_size=parameters.get('batch_size', 32)
    validation_split=parameters.get('validation_split', 0.1)
    shuffle=parameters.get('shuffle', False)

    # 2. LSTM Autoencoder 모델링
    model = Sequential([
        # Encoder
        LSTM(64, activation='relu', input_shape=(seq_len, 1), return_sequences=False),
        # 디코더에 입력하기 위해 시퀀스 길이만큼 반복
        RepeatVector(seq_len),
        # Decoder
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1)) # 각 타임스텝마다 1개의 출력
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    history = model.fit(
        X_train, X_train, # 입력과 출력이 동일 (오토인코더)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=shuffle
    )
