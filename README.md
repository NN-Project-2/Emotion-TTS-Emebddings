# EMOD: AN EFFICIENT APPROACH FOR LOW RESOURCE CONTROLLABLE EMOTIONAL SPEECH SYNTHESIS 🎤✨

## Overview
EMOD is a framework for enhancing expressive speech synthesis by capturing deep emotional embeddings from multilingual audio data. These embeddings integrate with end-to-end Text-to-Speech (TTS) models such as VITS and GPT-based speech models, enabling natural and controllable synthesis in low-resource language settings.

The extracted embeddings capture distinct emotions including happiness, sadness, anger, fear, surprise, and disgust, improving naturalness and human-likeness in generated speech. EMOD introduces fine-grained controllability, where deviations in pitch, energy, and duration are normalized into a continuous control factor. This parameter allows dynamic adjustment of emotional intensity, ranging from subtle expressivity to strong exaggeration.


### 🚀 **DEMO:** [Emotion-TTS Web](https://nn-project-2.github.io/Emotion-TTS-web/)
### 🎵 **Embeddings:** [Download Emotion embeddings.tar.xz](https://github.com/NN-Project-2/Emotion-TTS-Emebddings/blob/main/Emotion%20embeddings.tar.xz)

## 1. Table of Contents
1. [Introduction](#2-introduction)
2. [Emotional Embedding Database](#3-emotional-embedding-database)
3. [Integration with End-to-End TTS](#4-integration-with-e2e-tts)
4. [ Unsupervised Emotional Intensity Control](#5-how-the-intensity-unsupervised-was-trained-and-tuned)
5. [Emotional Speech Synthesis Model - Loss Functions](#6-emotional-speech-synthesis-model---loss-functions)
    1. [Mean Squared Error (MSE) Loss](#61-mean-squared-error-mse-loss-l_mse)
    2. [Generalized End-to-End (GE2E) Loss](#62-generalized-end-to-end-ge2e-loss-l_ge2e)
    3. [Cross-Entropy (CE) Loss](#63-cross-entropy-ce-loss-l_ce)
    4. [Orthogonality Loss](#64-orthogonality-loss-l_orth)
6. [Clustering for Emotion Cloning and Distance-Based Similarity](#7-clustering-for-emotion-cloning-and-distance-based-similarity)


<p align="center">
  <img src="Architecture/1Emod.png" alt="EMOD Architecture">
</p>

## 2. Introduction
The objective of EMOD is to develop an efficient **emotional embedding extractor** capable of capturing deep emotional features from **multilingual audio datasets** and integrating them with **TTS models**. It synthesizes speech conveying distinct emotions while preserving speaker identity.

The embeddings are **language-independent**, allowing transfer of emotional tones to new speakers in low-resource languages. EMOD handles diverse datasets, ensuring **consistent and expressive speech synthesis** across languages and speakers.


## 3. Emotional Embedding Database  

We curated a **multilingual audio database** with diverse emotions and speaker variations to train our emotion embedding extractor. This ensures **robust embeddings** capable of zero-shot generalization and fine-grained emotion control.  

### Key Highlights  

- Languages: Tamil, Malayalam, Hindi, English, Kannada, Telugu, with extra data from Assamese, Marathi, and Punjabi.  
- Emotions: Neutral, Angry, Sad, Happy, Fear, Surprise, Disgust.  
- Audio: 16 kHz `.wav` format, Mel spectrograms.  
- Speakers: Diverse male and female voices across ages.  
- Duration: ~10–20 hours per language.  
- Annotations: Emotion, speaker, language, intensity.  

Full dataset details are available [here](https://github.com/NN-Project-1/dis-Vector-Embedding/blob/main/README_1.md).

## 4. Integration with End-to-End TTS

The extracted **emotional embeddings** are integrated into **VITS** to condition speech synthesis while preserving speaker identity. In this architecture, the **Content Encoder** extracts speaker- and prosody-invariant linguistic features from input text, while the **Emotion Encoder** processes pitch (F₀), energy, duration, and timbre to generate emotion embeddings. These embeddings are projected into a **latent space (zemo)** with **speaker disentanglement**, allowing emotion to be applied independently. Emotional intensity is controlled via a **global scalar α**, which scales the overall embedding, and a **dimension-wise vector r**, which allows fine-grained adjustment of individual features. Latent embeddings can be interpolated across multiple emotions using weighted coefficients (λi) to create smooth transitions or composite emotional states. During training, the pre-trained Emotion Encoder is frozen, and the VITS components—including the flow-based prior, variance adaptor, and decoder—learn to reconstruct spectrograms while incorporating the emotional embeddings. Feature extraction ensures that F₀, energy, and duration are normalized relative to speaker-specific baselines, providing continuous control of intensity in a unified latent space.

<p align="center">
  <img src="Architecture/2o.png" alt="EMOD Architecture" width=400>
</p>


In the **GPT-based TTS pipeline**, input text is tokenized with a **BPE tokenizer** and embedded into subword representations, which are passed through **GPT-style Transformer blocks** trained to predict discrete latent codes from a **VQ-VAE encoder** of acoustic features. Pre-computed emotion embeddings, encoding both speaker identity and emotional state, are projected to match model dimensions and injected into the Transformer blocks via **concatenation and FiLM conditioning**. The same **α scalar** and dimension-wise ** vector** are applied to control the overall intensity and fine-grained aspects of emotion, allowing dynamic modulation of expressivity during synthesis. During training, emotion embeddings are incorporated into the model alongside content features, enabling the decoder to generate speech that reflects both the desired linguistic content and the specified emotional intensity, while maintaining speaker characteristics across multiple languages and speakers.

<p align="center">
  <img src="Architecture/e.png" alt="GPT Architecture" width=400>
</p>


## 5. Unsupervised Emotional Intensity Control


<p align="center">
  <img src="Architecture/i.png" alt="EMOD Architecture" width=300>
</p>


In our framework, emotional intensity was trained in an unsupervised manner by modeling deviations in prosodic cues relative to each speaker’s neutral baseline. Specifically, variations in pitch (∆F₀), energy (∆E), and duration (∆D) were extracted for every utterance and normalized to define a continuous intensity scalar α. During training, the Emotion Intensity Predictor learned to map these deviations into latent embeddings, enabling smooth control across weak to strong expressivity levels. At inference, α was directly applied to scale the emotional embedding globally, while a dimension-wise vector r adjusted fine-grained intensity per feature dimension. This design allowed natural tuning of emotional strength without requiring explicit intensity labels, supporting zero-shot transfer and controllable synthesis across multiple languages and speakers.



## 6. Emotional Speech Synthesis Model - Loss Functions
Our training process employs four key loss functions to optimize the emotional speech synthesis model effectively. These losses ensure accurate reconstruction, proper emotion classification, speaker discrimination, and disentanglement of speaker and emotion embeddings.

### 6.1. Mean Squared Error (MSE) Loss (L_MSE)
The Mean Squared Error (MSE) Loss is utilized to measure reconstruction accuracy. This loss function minimizes the difference between the original speech signal and its reconstructed version. By reducing reconstruction errors over samples, L_MSE ensures high-quality speech synthesis. The reconstructed spectrogram closely resembles the ground truth, maintaining intelligibility and expressiveness.

<p align="center">
  <img src="loss/mse.png" alt="EMOD Architecture" width=200>
</p>

### 6.2. Generalized End-to-End (GE2E) Loss (L_GE2E)
The Generalized End-to-End (GE2E) Loss is crucial for preserving speaker identity. It maximizes intra-speaker similarity while minimizing inter-speaker similarity, thereby improving speaker discrimination. By clustering embeddings from the same speaker closer together and pushing different speaker embeddings apart, GE2E loss effectively maintains speaker individuality during emotion transfer, ensuring that the synthesized speech retains the original speaker's characteristics.

<p align="center">
  <img src="loss/ge2e.png" alt="EMOD Architecture" width=200>
</p>

### 6.3. Cross-Entropy (CE) Loss (L_CE)
The Cross-Entropy (CE) Loss is applied to both the emotion classifier and the speaker classifier.
- For emotion classification, CE loss ensures that the extracted emotional features are accurately mapped to their corresponding emotion labels.
- In the speaker classification task, CE loss enforces correct speaker identity prediction. The adversarial training setup between emotion and speaker classifiers refines the model’s ability to distinguish between these aspects while improving robustness against unwanted biases.

<p align="center">
  <img src="loss/CE.png" alt="EMOD Architecture" width=200>
</p>

### 6.4. Orthogonality Loss (L_orth)
The Orthogonality Loss is introduced to disentangle emotion and speaker embeddings effectively. This loss function enforces orthogonality between the emotion and speaker representation spaces, preventing unwanted correlations. By ensuring that the extracted features from the emotion encoder do not overlap with speaker identity features, L_orth enhances the transferability of emotional embeddings across different speakers, facilitating effective cross-lingual and cross-gender emotion transfer.

<p align="center">
  <img src="loss/orth.png" alt="EMOD Architecture" width=200>
</p>

These four loss functions collectively optimize our model to achieve high-quality emotional speech synthesis while preserving speaker identity and ensuring accurate emotion representation. The integration of these loss mechanisms enables a robust zero-shot emotional TTS system adaptable to low-resource languages and diverse speaker conditions.

### Benefits:
- Prevents **speaker leakage** into the emotion embedding.
- Ensures that **emotion embedding** only captures emotional content.
- Guarantees better generalization in multi-speaker scenarios.

## 7. Clustering for Emotion Cloning and Distance-Based Similarity
To achieve high-fidelity emotion cloning, we utilize distance-based clustering to measure the similarity between emotional embeddings. We apply hierarchical clustering and K-means clustering on extracted emotion embeddings to group similar emotional states while preserving speaker identity. The similarity between a neutral speech sample and an emotional target is computed using cosine similarity and Euclidean distance in the embedding space. This ensures that cloned emotional speech retains the target emotion while maintaining the original speaker's characteristics. Additionally, a contrastive loss function is used to enhance intra-class clustering (same emotion) and increase inter-class separation (different emotions), further refining the accuracy of emotion cloning.

<p align="center">
  <img src="Architecture/cluster.png" alt="EMOD Architecture" width=300>
</p>


## 8. Zero-Shot Emotion Transfer and Control Scenarios in TTS  

- ✅ Emotion TTS  
- ✅ Cross-Lingual Transfer  
- ✅ Cross-Gender Emotion Transfer  
- ✅ Emotion Intensity Control  
- ✅ Integration with End-to-End TTS  


For more details, refer to the documentation. 🚀
