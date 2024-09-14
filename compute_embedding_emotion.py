import os
import torch
from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from TTS.utils.downloaders import download_vctk
from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.data import get_length_balancer_weights
from TTS.tts.utils.languages import LanguageManager, get_language_balancer_weights
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_balancer_weights, get_speaker_manager
torch.set_num_threads(8)

OUT_PATH = "/content/drive/MyDrive/emotion_emebdding_male_female/Embedding" 
MANIFEST_FOLDER = "/content/drive/MyDrive/emotion_emebdding_male_female/manifests"
EMO_VECTOR_FILES = []

for filename in tsv_files:
    parts = filename.replace('.csv', '').split('_')
    language = parts[0]  
    emotion = '_'.join(parts[1:]) 
    meta_file_train = os.path.join(MANIFEST_FOLDER, filename)
    dataset_config = BaseDatasetConfig(
        formatter="Emotion_TTS",
        meta_file_train=filename,  
        path=OUT_PATH,
        language=language,
    )

    embeddings_folder = os.path.join(OUT_PATH, f"{language}_{emotion}")
    os.makedirs(embeddings_folder, exist_ok=True)
    embeddings_file = os.path.join(embeddings_folder, "speakers.pth")
    
    if not os.path.isfile(embeddings_file):
        print("embeddings_folder:", embeddings_folder)
        print("meta_file_train:", meta_file_train)  
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_config.formatter,
            dataset_name=dataset_config.dataset_name,
            dataset_path=dataset_config.path,
            meta_file_train=meta_file_train,  
            meta_file_val=dataset_config.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    DATASETS_CONFIG_LIST.append(dataset_config)
    EMO_VECTOR_FILES.append(embeddings_file)

for dataset_config in DATASETS_CONFIG_LIST:
    print(f"Language: {dataset_config.language}, TSV File: {dataset_config.meta_file_train}")

print('DATASETS_CONFIG_LIST:::', DATASETS_CONFIG_LIST)


