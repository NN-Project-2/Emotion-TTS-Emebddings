import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.W_out(attn_output)

        return self.norm(attn_output + query)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dropout=0.1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.conv(x))))

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]

class EmotionEncoder(nn.Module):
    def __init__(self):
        super(EmotionEncoder, self).__init__()
        self.cnn = nn.Sequential(
            CNNBlock(80, 128, 3),
            CNNBlock(128, 256, 3),
            CNNBlock(256, 256, 3)
        )
        self.lstm = LSTMBlock(256, 128)
        self.fc = nn.Linear(256, 512)

    def forward(self, x):
        x = self.cnn(x)  
        x = x.permute(0, 2, 1)  
        x = self.lstm(x)
        embedding = self.fc(x[:, -1, :])  
        return x, embedding


class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()
        self.cnn = nn.Sequential(
            CNNBlock(768, 512, 3),
            CNNBlock(512, 256, 3),
            CNNBlock(256, 256, 3)
        )
        self.lstm = LSTMBlock(256, 128)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  
        x = self.lstm(x)
        return x


class SpeakerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_speakers):
        super(SpeakerClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers)
        )

    def forward(self, x):
        return self.classifier(x)

class EmotionClassifier(nn.Module):
    def __init__(self, embedding_dim, num_emotions):
        super(EmotionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_emotions)
        )

    def forward(self, x):
        return self.classifier(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = LSTMBlock(256, 128)
        self.cnn = nn.Sequential(
            CNNBlock(128, 192, 3),
            CNNBlock(192, 192, 3),
            nn.Conv1d(192, 80, 3, padding=1)
        )

    def forward(self, x):
        x = self.lstm(x)
        x = x.permute(0, 2, 1)
        return self.cnn(x)

class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
        self.cnn = nn.Sequential(
            CNNBlock(80, 128, 3),
            CNNBlock(128, 256, 3),
            CNNBlock(256, 80, 3)
        )

    def forward(self, x):
        return self.cnn(x) + x


class EmotionModel(nn.Module):
    def __init__(self, num_speakers, num_emotions):
        super(EmotionModel, self).__init__()
        # Separate Encoders
        self.emotion_encoder = EmotionEncoder()
        self.content_encoder = ContentEncoder()
        self.cross_attention = CrossAttention(embed_dim=256, num_heads=8)
        self.decoder = Decoder()
        self.postnet = PostNet()

        # Classifiers
        self.speaker_classifier = SpeakerClassifier(512, num_speakers)
        self.emotion_classifier = EmotionClassifier(512, num_emotions)

    def forward(self, mel_spectrogram, hubert_content):
        """
        mel_spectrogram: [B, 80, T]
        hubert_content: [B, T, 768]
        """
        # Emotion Encoder
        emotion_encoded, emotion_embedding = self.emotion_encoder(mel_spectrogram)

        # Content Encoder (HubERT content)
        content_encoded = self.content_encoder(hubert_content)

        # Cross Attention to fuse Content & Emotion
        fused_features = self.cross_attention(emotion_encoded, content_encoded, content_encoded)

        # Decoder
        mel_reconstructed = self.decoder(fused_features)
        mel_refined = self.postnet(mel_reconstructed)

        # Classification
        speaker_pred = self.speaker_classifier(emotion_embedding)
        emotion_pred = self.emotion_classifier(emotion_embedding)

        return mel_reconstructed, mel_refined, emotion_embedding, speaker_pred, emotion_pred
