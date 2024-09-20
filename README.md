## EMOD: An efficient approach for low resource emotional speech synthesis 🎤✨

## Overview
This repository contains the code and resources for developing emotional embeddings to enhance speech synthesis. Our focus is on creating embeddings that capture various emotions in multilingual contexts, improving the expressiveness of text-to-speech systems.

DEMO:https://nn-project-2.github.io/Emotion-TTS-web/

Embeddings: A sample embedding file for emotion conversion is available. You can download it here: [Emotion embeddings.tar.xz.](https://github.com/NN-Project-2/Emotion-TTS-Emebddings/blob/main/Emotion%20embeddings.tar.xz)

## 📚 Table of Contents
- [Introduction](#introduction)
- [Emotional Embedding Database](#emotional-embedding-database)
- [Integration with E2E TTS](#integration-with-e2e-tts)
- [Zero-Shot Cloning with VITS](#zero-shot-cloning-with-vits)
- [Experimental Results](#experimental-results)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [License](#license)

## Introduction
Our model integrates emotional embeddings into the VITS TTS architecture, allowing for emotionally expressive speech synthesis. Supported emotions include:

- 😠 **Anger**
- 😢 **Sadness**
- 😲 **Surprise**
- 😊 **Happiness**
- 😨 **Fear**
- 🤢 **Disgust**

## Emotional Embedding Database
We constructed a custom emotional embedding database sourced from dramas and stories, featuring contributions from male and female actors across multiple languages. Key highlights:

- **Languages**: Tamil, Malayalam, Hindi, English, German
- **Balanced Samples**: Approximately 30 minutes of audio data per emotion
- **Quality**: Down-sampled to 16 kHz, with Mel spectrograms extracted

The dataset is meticulously organized by language, speaker, and gender, enhancing the model's ability to capture diverse emotional expressions.

## Integration with E2E TTS
We utilize the VITS architecture to integrate emotional embeddings effectively. The process involves:

1. **Speaker Embedding Extraction**: Capturing unique vocal characteristics.
2. **Emotion Embedding Retrieval**: Mapping emotional attributes from our database.
3. **Superimposition of Emotion**: Merging embeddings to generate expressive speech.

## Zero-Shot Cloning with VITS
Our zero-shot cloning approach employs clustering techniques to synthesize speech for new target voices. Key steps include:

- **Speaker Embedding Extraction**
- **Emotion Embedding Retrieval**
- **Feature Mapping and Clustering**
- **Superimposition of Emotion**: Using element-wise addition for embedding integration.

## Experimental Results
We evaluated our model using two primary datasets: LIMMITS and an Emotional Dataset. Results indicate strong performance across various languages and emotions:

| Language  | Emotion  | Similarity (%) | Classifier Accuracy (%) | MOS  |
|-----------|----------|----------------|-------------------------|------|
| English   | Angry    | 89             | 85                      | 3.81 |
| Hindi     | Sad      | 76             | 81                      | 3.72 |
| Malayalam | Happy    | 79             | 83                      | 3.61 |
| Tamil     | Angry    | 83             | 79                      | 3.75 |


## Installation
Clone the repository and install the required packages:

## Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/repo.git
cd repo
pip install -r requirements.txt

## 🏁 Conclusion
The emotion embedding database plays a crucial role in enhancing the emotional expressiveness of synthesized speech. By incorporating a diverse collection of audio recordings spanning multiple languages and emotional states, we have developed a comprehensive dataset that supports robust emotion embedding extraction. This dataset ensures that our models can accurately capture and convey a wide range of emotions across different languages. The inclusion of zero-shot emotion conversion capabilities in our TTS system further advances the flexibility of emotional expression in synthesized speech. By utilizing these emotion embeddings, the TTS system can generate speech with new emotional tones not seen during training, broadening its applicability. The synthesized emotional speech outputs can be evaluated using a classifier model, which is trained to assess the accuracy of emotional expression, ensuring that the speech aligns with the intended emotional state. The results and detailed embeddings are provided in the accompanying Git repository. 

For more details, please refer to the documentation in this repository! Happy experimenting! 🚀


