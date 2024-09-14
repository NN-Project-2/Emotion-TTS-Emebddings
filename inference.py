import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from TTS.Emotion_TTS_conversion.data.dataset import prepare_data, load_emotion_and_language_maps
from TTS.Emotion_TTS_conversion.data_loader import EmotionDataset
from TTS.Emotion_TTS_conversion.model.EmotionModel import EmotionModel
from TTS.Emotion_TTS_conversion.train import train_model
import librosa
from TTS.Emotion_TTS_conversion.data.dataset import process_audio

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def preprocessing(config):
    """Perform data preprocessing."""
    emotion_map, language_map = load_emotion_and_language_maps(config['emoiton_ids'], config['language_ids'])
    print("Starting data preparation...")
    prepare_data(
        config['data_dir'], config['processed_folder'], emotion_map, language_map, 
        sample_rate=config['sample_rate'], n_mfcc=config['n_mfcc']
    )
    print("Data preparation complete!")

def inference(wav_file):
    """Perform inference on a single WAV file."""
    print(f"Processing WAV file: {wav_file}")
    config='/usr/local/lib/python3.10/dist-packages/TTS/Emotion_TTS_conversion/config/config.json'
    features = process_audio(wav_file, 
                             sample_rate=config.get('sample_rate', 16000), 
                             n_mfcc=config.get('n_mfcc', 80), 
                             max_length=config.get('max_length', 192))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    model = EmotionModel().to(device)
    
    checkpoint_path = config['checkpoint_path']
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    else:
        print(f"No checkpoint found at {checkpoint_path}.")
        return
    print("Starting inference...")
    with torch.no_grad():
        latent, embedding, _ = model(features)
        print("Latent shape:", latent.shape)
        print("Embedding shape:", embedding.shape)
    
    return embedding.cpu().numpy()

