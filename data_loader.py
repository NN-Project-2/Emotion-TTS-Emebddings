import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, processed_folder):
        self.data = np.load(os.path.join(processed_folder, "dataset.npy"), allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_path, speaker_id, emotion_id = self.data[idx]

        # Load features
        mel_spectrogram = np.load(os.path.join(feature_path, "mel_spectrogram.npy"))
        pitch = np.load(os.path.join(feature_path, "pitch.npy"))
        energy = np.load(os.path.join(feature_path, "energy.npy"))
        hubert = np.load(os.path.join(feature_path, "hubert.npy"))

        # Convert IDs
        speaker_id = int(speaker_id)
        emotion_id = int(emotion_id)

        return (torch.tensor(mel_spectrogram, dtype=torch.float32),
                torch.tensor(pitch, dtype=torch.float32),
                torch.tensor(energy, dtype=torch.float32),
                torch.tensor(hubert, dtype=torch.float32),
                torch.tensor(speaker_id, dtype=torch.long),
                torch.tensor(emotion_id, dtype=torch.long))

def collate_fn(batch):
    mel_spectrograms, pitches, energies, huberts, speaker_ids, emotion_ids = zip(*batch)

    mel_spectrograms = torch.nn.utils.rnn.pad_sequence(mel_spectrograms, batch_first=True, padding_value=0)
    pitches = torch.nn.utils.rnn.pad_sequence(pitches, batch_first=True, padding_value=0)
    energies = torch.nn.utils.rnn.pad_sequence(energies, batch_first=True, padding_value=0)
    huberts = torch.nn.utils.rnn.pad_sequence(huberts, batch_first=True, padding_value=0)

    speaker_ids = torch.stack(speaker_ids)
    emotion_ids = torch.stack(emotion_ids)

    return mel_spectrograms, pitches, energies, huberts, speaker_ids, emotion_ids
