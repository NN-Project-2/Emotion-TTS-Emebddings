import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from data.dataset import prepare_data, load_emotion_and_language_maps, process_audio
from data_loader import EmotionDataset
from model.EmotionModel import EmotionModel
from trainer import Trainer  

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_data(config):
    emotion_map, language_map = load_emotion_and_language_maps(config['emotion_ids'], config['language_ids'])
    prepare_data(config['data_dir'], config['processed_folder'], emotion_map, language_map, 
                 sample_rate=config['sample_rate'], n_mfcc=config['n_mfcc'])

def train_emotion_model(config):
    dataset = EmotionDataset(config['processed_folder'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionModel().to(device)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    trainer = Trainer(model, dataloader, config['num_epochs'], config['learning_rate'], config['checkpoint_dir'], device)
    trainer.train()

def run_inference(config, wav_file):
    features = process_audio(wav_file, sample_rate=config.get('sample_rate', 16000), 
                             n_mfcc=config.get('n_mfcc', 80), max_length=config.get('max_length', 192))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    model = EmotionModel().to(device)
    
    checkpoint_path = config.get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    else:
        print(f"No checkpoint found at {checkpoint_path}.")
        return

    with torch.no_grad():
        mel_reconstructed, mel_refined, emotion_embedding = model(features)
    
    return emotion_embedding.cpu().numpy()

def main(config, mode, wav_file=None):
    if mode == 1:
        preprocess_data(config)
    elif mode == 2:
        train_emotion_model(config)
    elif mode == 3:
        if wav_file:
            embedding = run_inference(config, wav_file)
            if embedding is not None:
                print("Extracted Emotion Embedding:", embedding)
        else:
            print("Please provide a WAV file for inference.")
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=int, choices=[1, 2, 3])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--wav_file', type=str)

    args = parser.parse_args()
    config = load_config(args.config)
    main(config, args.mode, args.wav_file)
