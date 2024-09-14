import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from data.dataset import prepare_data, load_emotion_and_language_maps
from data_loader import EmotionDataset
from model.EmotionModel import EmotionModel
from train import train_model
import librosa
from data.dataset import process_audio

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def preprocessing(config):
    """Perform data preprocessing."""
    # Load emotion and language mappings
    emotion_map, language_map = load_emotion_and_language_maps(config['emoiton_ids'], config['language_ids'])
    
    # Data preparation
    print("Starting data preparation...")
    prepare_data(
        config['data_dir'], config['processed_folder'], emotion_map, language_map, 
        sample_rate=config['sample_rate'], n_mfcc=config['n_mfcc']
    )
    print("Data preparation complete!")

def training(config):
    """Train the model."""
    # Load dataset
    print("Loading dataset...")
    dataset = EmotionDataset(config['processed_folder'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    model = EmotionModel().to(device)
    
    # Ensure checkpoint directory exists
    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])
    
    # Train the model
    print("Starting training...")
    train_model(
        model, dataloader, config['num_epochs'], config['learning_rate'], 
        config['checkpoint_dir'], device
    )
    print("Training complete!")


def inference(config, wav_file):
    """Perform inference on a single WAV file."""
    print(f"Processing WAV file: {wav_file}")
    
    # Process the audio file to extract features
    features = process_audio(wav_file, 
                             sample_rate=config.get('sample_rate', 16000), 
                             n_mfcc=config.get('n_mfcc', 80), 
                             max_length=config.get('max_length', 192))

    # Convert features to tensor and add batch dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Initialize model and load checkpoint
    model = EmotionModel().to(device)
    
    checkpoint_path = config['checkpoint_path']
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    else:
        print(f"No checkpoint found at {checkpoint_path}.")
        return

    # Perform inference
    print("Starting inference...")
    with torch.no_grad():
        latent, embedding, _ = model(features)
        print("Latent shape:", latent.shape)
        print("Embedding shape:", embedding.shape)
    
    # Optionally return or save the embedding
    return embedding.cpu().numpy()


def main(config, mode, wav_file=None):
    """Main entry point for the script."""
    if mode == 1:
        preprocessing(config)
    elif mode == 2:
        training(config)
    elif mode == 3:
        if wav_file is None:
            print("Please provide a WAV file for inference.")
        else:
            inference(config, wav_file)
    else:
        print("Invalid mode selected. Choose 1 for preprocessing, 2 for training, or 3 for inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Embedding Model Pipeline")
    parser.add_argument('mode', type=int, choices=[1, 2, 3], help="1: Preprocessing, 2: Training, 3: Inference")
    parser.add_argument('--config', type=str, default="/content/drive/MyDrive/Emotion_Model_Training_Finetuning/Emo_Emb_Model/config/config.json", help="Path to the config file")
    parser.add_argument('--wav_file', type=str, help="Path to the WAV file for inference")
    
    args = parser.parse_args()

    config = load_config(args.config)
    main(config, args.mode, args.wav_file)
