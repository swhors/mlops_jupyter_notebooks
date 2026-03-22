import torch
import torch.nn as nn


# 2. LSTM Autoencoder 모델 정의
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        # Encoder: 입력 시퀀스를 숨겨진 상태(hidden state)로 압축
        self.encoder = nn.LSTM(n_features, embedding_dim, batch_first=True)
        
        # Decoder: 압축된 상태에서 시퀀스 복원
        self.decoder = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.output_layer = nn.Linear(embedding_dim, n_features)
        
    def forward(self, x):
        # 인코딩
        _, (hidden, _) = self.encoder(x)
        # hidden shape: (1, batch_size, embedding_dim)
        
        # 디코더 입력 생성 (인코더의 마지막 hidden state를 반복하여 사용)
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        
        # 디코딩
        decoder_out, _ = self.decoder(hidden)
        
        # 최종 출력 레이어
        output = self.output_layer(decoder_out)
        return output