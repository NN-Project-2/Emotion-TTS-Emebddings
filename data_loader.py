import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, processed_folder):
        self.data = np.load(os.path.join(processed_folder, "dataset.npy"), allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_path, language_id, emotion_id = self.data[idx]
        
        # Load the features
        features = np.load(feature_path)
        
        # Convert to appropriate types
        features = features.astype(np.float32)
        language_id = int(language_id) 
        emotion_id = int(emotion_id)    
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(language_id, dtype=torch.long), torch.tensor(emotion_id, dtype=torch.long)
