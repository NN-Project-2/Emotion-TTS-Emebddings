import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)
        
class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
    def forward(self, x):
        return self.lstm(x)[0]

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.encoder_cnn = nn.Sequential(
            CNNBlock(192, 256, kernel_size=3),
            CNNBlock(256, 256, kernel_size=3),
            CNNBlock(256, 256, kernel_size=3),
            CNNBlock(256, 256, kernel_size=3),
            CNNBlock(256, 256, kernel_size=3)
        )
        self.encoder_lstm = LSTMBlock(input_size=256, hidden_size=128, num_layers=3, batch_first=False)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_embedding = nn.Linear(128 * 233, 512)  
        self.latent_channels = 128
        self.latent_length = 233  
        self.decoder_lstm = LSTMBlock(input_size=self.latent_channels, hidden_size=128, num_layers=3, batch_first=False)
        
        self.decoder_cnn = nn.Sequential(
            CNNBlock(128, 192, kernel_size=3, padding=1), 
            nn.Conv1d(192, 192, kernel_size=3, padding=1), 
            nn.Conv1d(192, 192, kernel_size=3, padding=1), 
            nn.Conv1d(192, 192, kernel_size=3, padding=1)  
        )
    
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = x.permute(0, 2, 1) 
        latent = self.encoder_lstm(x)
        latent = latent.permute(0, 2, 1) 
        latent_flat = self.flatten(latent)
        embedding = self.fc_embedding(latent_flat)
        decode = latent.permute(0, 2, 1)  
        decode = self.decoder_lstm(decode)
        decode = decode.permute(0, 2, 1)  
        reconstruction = self.decoder_cnn(decode)
        return latent, embedding, reconstruction
