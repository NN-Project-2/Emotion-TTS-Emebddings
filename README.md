# EMOD: AN EFFICIENT APPROACH FOR LOW RESOURCE CONTROLLABLE EMOTIONAL SPEECH SYNTHESIS 🎤✨

## Overview
EMOD is a framework for enhancing expressive speech synthesis by capturing deep emotional embeddings from multilingual audio data. These embeddings integrate with end-to-end Text-to-Speech (TTS) models such as VITS and GPT-based speech models, enabling natural and controllable synthesis in low-resource language settings.

The extracted embeddings capture distinct emotions including happiness, sadness, anger, fear, surprise, and disgust, improving naturalness and human-likeness in generated speech. EMOD introduces fine-grained controllability, where deviations in pitch, energy, and duration are normalized into a continuous control factor. This parameter allows dynamic adjustment of emotional intensity, ranging from subtle expressivity to strong exaggeration.


### 🚀 **DEMO:** [Emotion-TTS Web](https://nn-project-2.github.io/Emotion-TTS-web/)
### 🎵 **Embeddings:** [Download Emotion embeddings.tar.xz](https://github.com/NN-Project-2/Emotion-TTS-Emebddings/blob/main/Emotion%20embeddings.tar.xz)

## 1. Table of Contents
1. [Overview](#overview)
2. [Introduction](#2-introduction)
    - [2.1 Intensity Control Parameter α](#21-intensity-control-parameter-α)
3. [Integration with End-to-End TTS](#3-integration-with-end-to-end-tts)
    - [3.1 VITS Architecture](#31-vits-architecture)
    - [3.2 GPT-Based TTS Architecture](#32-gpt-based-tts-architecture)
4. [Unsupervised Emotional Intensity Control](#4-unsupervised-emotional-intensity-control)
5. [Emotional Speech Synthesis Model – Loss Functions](#5-emotional-speech-synthesis-model--loss-functions)
    - [Mean Squared Error (MSE) Loss](#mean-squared-error-mse-loss-l_mse)
    - [Generalized End-to-End (GE2E) Loss](#generalized-end-to-end-ge2e-loss-l_ge2e)
    - [Cross-Entropy (CE) Loss](#cross-entropy-ce-loss-l_ce)
    - [Orthogonality Loss](#orthogonality-loss-l_orth)
6. [Clustering for Emotion Cloning and Distance-Based Similarity](#6-clustering-for-emotion-cloning-and-distance-based-similarity)
7. [Test Setup and Results](#7-test-setup-and-results)
    - [7.1 Test Setup](#71-test-setup)
    - [7.2 Results Across Languages](#72-results-across-languages)
    - [7.3 Emotion Transfer and Speaker Cloning Results](#73-emotion-transfer-and-speaker-cloning-results)
8. [Emotional Embedding Database](#8-emotional-embedding-database)
9. [Zero-Shot Emotion Transfer and Control Scenarios in TTS](#9-zero-shot-emotion-transfer-and-control-scenarios-in-tts)




<p align="center">
  <img src="Architecture/1Emod.png" alt="EMOD Architecture">
</p>

## 2. Introduction

EMOD is designed as a multilingual emotional embedding extractor that enables controllable emotional speech synthesis within both VITS and GPT-based text-to-speech (TTS) architectures. The system models six categorical emotions  happiness, sadness, anger, fear, surprise, and disgust using language-independent 256-dimensional emotion embeddings trained with L2 normalization and contrastive objectives. These embeddings are constructed to remain stable across languages while preserving perceptually salient emotional characteristics, allowing direct reuse in zero-shot and low-resource synthesis settings.

The architecture consists of three tightly coupled components. A transformer-based encoder processes prosodic and spectral cues derived from 80-band mel-spectrograms, including fundamental frequency (F₀), energy contours, and temporal structure, to form a compact emotional representation. An intensity control module models continuous variations in emotional strength directly from prosodic deviations rather than discrete labels. Finally, a TTS integration layer adapts the learned emotion embeddings for both VITS and GPT-based backends through feature-wise linear modulation (FiLM), ensuring compatibility with structurally different synthesis models without retraining the emotion encoder.

Training follows a three-stage pipeline designed to balance cross-lingual generalization and representation disentanglement. Multilingual pre-training is first performed using speech from Tamil, Malayalam, Hindi, English, Kannada, and Telugu, supplemented with Assamese, Marathi, and Punjabi, enabling the model to learn language-agnostic emotional cues. This is followed by a disentanglement phase that enforces orthogonality between emotion and speaker representations while employing adversarial speaker classification to suppress speaker leakage. The final stage fine-tunes the embeddings within TTS-specific objectives to ensure stable conditioning during synthesis. All models are trained for 500,000 steps using AdamW with a learning rate of 3×10⁻⁴, batch size 32, gradient accumulation of two steps, and mixed-precision training on four NVIDIA L4 GPUs (24 GB VRAM each). The training data comprises approximately 10–20 hours of speech per language at 16 kHz.

### 2.1 Intensity Control Parameter α

Although categorical emotion embeddings encode emotional type, they do not capture how strongly an emotion is expressed. In natural speech, emotional expression varies continuously, even within the same category, depending on context, speaker intent, and prosodic realization. EMOD explicitly models this variation using a continuous global intensity parameter α ∈ [0.0, 1.0], which separates emotional magnitude from emotional identity. This separation enables controlled adjustment of expressivity without altering the semantic direction of the emotion embedding.

The parameter α is computed from low-level prosodic deviations relative to speaker-specific neutral baselines, ensuring invariance to individual speaking styles. For each utterance, deviations in pitch (ΔF₀), energy (ΔE), and duration (ΔD) are measured against neutral reference statistics for the same speaker. These deviations reflect how far an utterance departs from neutral prosody along dimensions known to correlate with perceived emotional intensity. The deviations are linearly combined using learnable weights and normalized through a sigmoid function to produce a bounded intensity value:
<p align="center">
α = σ(β₁ΔF₀ + β₂ΔE + β₃ΔD)
</p>
where σ denotes the sigmoid function and βᵢ are trainable coefficients that adaptively weight the relative contribution of each prosodic cue. This formulation allows the model to emphasize the most informative dimensions while maintaining numerical stability and cross-speaker consistency.

During inference, α operates as a global scaling factor applied directly to the emotion embedding:
<p align="center">
z̃_emo = α · z_emo
</p>
This multiplicative interaction enables smooth interpolation between neutral and expressive speech while preserving the direction of the emotion vector in latent space. Because α is independent of emotion classification, it can be adjusted dynamically at synthesis time without re-encoding emotional category labels. The formulation remains compatible with dimension-wise modulation vectors r ∈ ℝ²⁵⁶, enabling joint control over global intensity and localized acoustic attributes.


## 3. Integration with End-to-End TTS

The extracted emotion embeddings are integrated into both **VITS** and **GPT-based TTS** architectures to enable controllable emotional synthesis while preserving linguistic content and speaker identity. The Emotion Encoder generates a continuous emotion representation $z_{emo}$ from prosodic and spectral cues, which is explicitly disentangled from content and speaker representations and projected into a latent space compatible with downstream synthesis models. Emotional strength is regulated using the global scalar $\alpha$, which modulates the magnitude of emotional deviation without altering emotional type:

<p align="center">
  $\tilde{z}_{emo} = \alpha \cdot (\mathbf{r} \odot z_{emo})$
</p>

Increasing $\alpha$ amplifies emotion-related attributes such as pitch variance, energy dynamics, and spectral coloration, while the direction of $z_{emo}$ preserves categorical emotion identity. This design enables monotonic and continuous intensity control, which is particularly important in zero-shot and cross-lingual settings where explicit intensity annotations are unavailable.

To enable finer control, a dimension-wise modulation vector \( r \) is applied alongside $\alpha$:

Here, \( r \) selectively scales latent subspaces corresponding to specific acoustic attributes, allowing independent adjustment of pitch, energy, or timbre while maintaining a coherent emotional representation.

---

## 3.1 VITS Architecture

In the VITS architecture, the scaled emotion embedding $z_{emo}$ conditions the flow-based prior and decoder while remaining disentangled from the Content Encoder, which extracts speaker- and prosody-invariant linguistic features. The variance adaptor normalizes $F_0$, energy, and duration relative to speaker-specific baselines, allowing $\alpha$ to directly modulate deviations from neutral prosody rather than absolute acoustic values. During training, the Emotion Encoder is frozen, and VITS components learn to reconstruct mel-spectrograms conditioned on $z_{emo}$, enabling precise and stable emotion control.


The effect of $\alpha$ is validated by performing **controlled inference experiments**, where $\alpha$ is varied while keeping content, speaker embedding, and random seed fixed. Objective evaluation measures monotonic changes in $F_0$ variance, energy range, and duration spread, confirming that $\alpha$ consistently amplifies expressivity without distorting content or speaker identity. Subjective evaluation is conducted using **Mean Opinion Score (MOS)** tests, where listeners rate emotional intensity and naturalness across different $\alpha$ values. Results show a perceptually linear and stable increase in emotional strength as $\alpha$ increases, with no artifacts or speaker leakage, confirming that $\alpha$ functions as a **reliable and interpretable control parameter** for VITS-based TTS.


<p align="center">
  <img src="Architecture/2o.png" alt="VITS Architecture" width=400>
</p>

---

## 3.2 GPT-Based TTS Architecture

In the **GPT-based TTS pipeline**, input text is tokenized using a **BPE tokenizer** and embedded into subword representations, which are processed by **GPT-style Transformer decoder blocks** trained to predict discrete acoustic tokens derived from a **VQ-VAE encoder**. The pre-computed emotion embeddings $z_{emo}$ are projected into the Transformer hidden dimension and injected using **concatenation and FiLM-based conditioning**. The scaled embedding $\tilde{z}\_{emo} = \alpha \cdot (\mathbf{r} \odot z\_{emo})$ is applied uniformly across all Transformer layers, enabling continuous modulation of expressivity during autoregressive token generation while preserving temporal coherence and speaker identity. $\alpha$ controls the **global emotional magnitude**, while $\mathbf{r}$ fine-tunes individual feature dimensions, allowing independent adjustment of pitch, energy, or timbre dynamics. Evaluation follows a similar methodology as in VITS: objective metrics track changes in prosodic statistics and speaker similarity, and subjective MOS tests quantify perceived emotional intensity and naturalness. Results confirm that $\alpha$ functions as a **stable, interpretable, and monotonic control parameter**, providing continuous, zero-shot, and cross-lingual controllable emotion synthesis in GPT-based TTS systems.

<p align="center">
  <img src="Architecture/e.png" alt="GPT Architecture" width=400>
</p>

## 4. Unsupervised Emotional Intensity Control


<p align="center">
  <img src="Architecture/i.png" alt="EMOD Architecture" width=300>
</p>


In our framework, emotional intensity was trained in an unsupervised manner by modeling deviations in prosodic cues relative to each speaker’s neutral baseline. Specifically, variations in pitch (∆F₀), energy (∆E), and duration (∆D) were extracted for every utterance and normalized to define a continuous intensity scalar α. During training, the Emotion Intensity Predictor learned to map these deviations into latent embeddings, enabling smooth control across weak to strong expressivity levels. At inference, α was directly applied to scale the emotional embedding globally, while a dimension-wise vector r adjusted fine-grained intensity per feature dimension. This design allowed natural tuning of emotional strength without requiring explicit intensity labels, supporting zero-shot transfer and controllable synthesis across multiple languages and speakers.


## 5. Emotional Speech Synthesis Model – Loss Functions

The emotional speech synthesis model is optimized using a set of complementary loss functions, each targeting a specific technical requirement of controllable emotional TTS. The combined objective ensures stable acoustic reconstruction, robust speaker preservation, accurate emotion encoding, and effective disentanglement between latent factors.

- **Mean Squared Error (MSE) Loss (L_MSE)**  
  Used to enforce accurate acoustic reconstruction by minimizing the frame-level error between predicted and ground-truth mel-spectrograms. This loss stabilizes training under emotion-driven prosodic variation and ensures that changes in emotional intensity do not degrade phonetic structure, spectral continuity, or speech intelligibility. It provides a strong low-level constraint that anchors higher-level emotion and speaker objectives to perceptually valid speech outputs.

  <p align="center">
    <img src="loss/mse.png" alt="MSE Loss" width="200">
  </p>

- **Generalized End-to-End (GE2E) Loss (L_GE2E)**  
  Applied to speaker embeddings to preserve speaker identity during emotional modulation. GE2E loss encourages tight clustering of embeddings belonging to the same speaker while maximizing separation across different speakers, thereby preventing speaker drift when emotion embeddings are injected. This is critical in zero-shot emotion transfer, where the emotional signal must modify prosody without contaminating speaker-specific timbre characteristics.

  <p align="center">
    <img src="loss/ge2e.png" alt="GE2E Loss" width="200">
  </p>

- **Cross-Entropy (CE) Loss (L_CE)**  
  Used for both emotion and speaker classification objectives to enforce discriminative latent representations. For emotion classification, CE loss ensures that emotional embeddings are linearly separable across emotion categories. For speaker classification, it is used in an adversarial setting to discourage speaker-identifiable information from leaking into the emotion embedding space, thereby improving robustness and disentanglement.

  <p align="center">
    <img src="loss/CE.png" alt="Cross-Entropy Loss" width="200">
  </p>

- **Orthogonality Loss (L_orth)**  
  Introduced to explicitly decouple emotion and speaker representations by enforcing orthogonality between their embedding subspaces. This loss minimizes correlation between emotion and speaker vectors, ensuring that emotional modulation remains transferable across unseen speakers, languages, and genders. It plays a crucial role in maintaining controllable emotion expression without compromising speaker identity.

  <p align="center">
    <img src="loss/orth.png" alt="Orthogonality Loss" width="200">
  </p>

Together, these loss components form a multi-objective optimization framework that enables high-quality emotional speech synthesis with precise control, strong disentanglement, and reliable generalization in low-resource and cross-lingual scenarios.



## 6. Clustering for Emotion Cloning and Distance-Based Similarity
To achieve high-fidelity emotion cloning, we utilize distance-based clustering to measure the similarity between emotional embeddings. We apply hierarchical clustering and K-means clustering on extracted emotion embeddings to group similar emotional states while preserving speaker identity. The similarity between a neutral speech sample and an emotional target is computed using cosine similarity and Euclidean distance in the embedding space. This ensures that cloned emotional speech retains the target emotion while maintaining the original speaker's characteristics. Additionally, a contrastive loss function is used to enhance intra-class clustering (same emotion) and increase inter-class separation (different emotions), further refining the accuracy of emotion cloning.

<p align="center">
  <img src="Architecture/cluster.png" alt="EMOD Architecture" width=300>
</p>

## 7. Test Setup and Results

This section describes the evaluation protocol used to assess the proposed emotional speech synthesis system, followed by a detailed analysis of quantitative and subjective results across multiple languages and speakers. The evaluation is designed to measure emotion correctness, speaker similarity, and perceptual naturalness under zero-shot and cross-lingual conditions.

---

### 7.1 Test Setup


The evaluation of the emotional speech synthesis system is designed to rigorously quantify emotion accuracy, speaker preservation, and perceptual naturalness across multiple languages and speakers. The testing protocol combines **objective SER-based classification**, **embedding similarity and clustering**, and **human perceptual ratings** to provide both technical and perceptual evidence of model performance.

For **emotion correctness**, a pretrained **Speech Emotion Recognition (SER) model, `wav2vec2-SER`**, is used to classify synthesized speech into one of the target emotion categories. The predicted labels are compared with the intended emotions to compute **classification accuracy (Cls. Acc.)**. This approach ensures that the emotional embeddings encode acoustically and perceptually distinguishable emotion features. The SER model provides a standardized, reproducible benchmark for emotion transfer in zero-shot and cross-lingual conditions.

To measure **emotion similarity**, embeddings are extracted from synthesized speech and compared against reference emotional speech embeddings using **cosine similarity**, reported as **Sim. (%)**. Clustering analysis is also performed on the embedding space to verify that synthesized samples group consistently with their corresponding emotion categories, confirming that the latent space is well-structured and that emotional distinctions are maintained across speakers and languages.

For **speaker preservation and perceptual quality**, **Mean Opinion Score (MOS)** evaluations are conducted with human listeners. Participants rate the synthesized samples on a 1–5 scale in three categories: **naturalness**, **speaker similarity** (for cloning), and **emotional expressiveness**. All evaluation samples are generated under **zero-shot conditions**, with unseen speakers and unseen emotional intensities, to test cross-speaker and cross-lingual generalization. This combination of objective SER metrics, embedding analysis, and subjective listening ensures a comprehensive, technically rigorous assessment of the model's controllable emotional speech synthesis capabilities.


---

### 7.2 Results Across Languages

The following table reports performance across four languages, measuring emotion embedding similarity (**Sim.**), SER-based emotion classification accuracy (**Cls. Acc.**), and perceptual naturalness (**MOS**).

| Language    | Emotion | Sim. (%) | Cls. Acc. (%) | MOS |
|------------|---------|----------|---------------|-----|
| English    | Angry   | 89       | 85            | 3.81 |
| Hindi      | Sad     | 76       | 81            | 3.72 |
| Malayalam  | Happy   | 79       | 83            | 3.61 |
| Tamil      | Angry   | 83       | 79            | 3.75 |

These results indicate strong emotion preservation across languages, with high similarity scores and consistent classification accuracy despite linguistic variation. The MOS values demonstrate that emotional modulation does not significantly degrade naturalness, even in low-resource languages such as Malayalam and Tamil. The relatively high similarity and accuracy scores validate the language-independent nature of the learned emotion embeddings.

---

### 7.3 Emotion Transfer and Speaker Cloning Results

To evaluate emotion transfer quality under speaker cloning conditions, MOS evaluations are conducted separately for **speaker similarity (Cloning)** and **emotional expressiveness (Emotion)** across different target speakers and emotions.

| Target Speaker    | Emotion  | MOS (Cloning) | MOS (Emotion) |
|------------------|----------|---------------|---------------|
| English Female   | Angry    | 3.61          | 3.63 |
| English Male     | Disgust  | 3.78          | 3.75 |
| Hindi Male       | Sad      | 3.56          | 3.54 |
| Hindi Female     | Fear     | 3.87          | 3.77 |
| Tamil Female     | Angry    | 3.76          | 3.69 |
| Tamil Male       | Angry    | 3.49          | 3.56 |
| Malayalam Male   | Happy    | 3.77          | 3.68 |
| Malayalam Female | Happy    | 3.68          | 3.52 |

The results show that the model effectively preserves speaker identity while transferring emotional attributes, as reflected by consistently balanced MOS scores for cloning and emotion. Minor variations across speakers can be attributed to differences in recording conditions and speaker-specific prosodic ranges. Overall, the close alignment between cloning and emotion MOS scores confirms that emotional modulation is achieved without introducing speaker distortion.

---

These evaluations collectively demonstrate that the proposed framework achieves robust emotional control, reliable emotion transfer, and high perceptual quality across languages and speakers, validating its suitability for zero-shot emotional TTS in low-resource and multilingual scenarios.


## 8. Emotional Embedding Database

A multilingual emotional speech database is curated to train the emotion embedding extractor with sufficient linguistic, emotional, and speaker variability. The objective of this dataset design is to learn language-independent and speaker-invariant emotional representations that generalize effectively in zero-shot and cross-lingual synthesis scenarios, while still supporting fine-grained emotion intensity control.

### Key Highlights

- **Languages:** Tamil, Malayalam, Hindi, English, Kannada, and Telugu, with additional supplementary data from Assamese, Marathi, and Punjabi to improve cross-lingual robustness.  
- **Emotion Categories:** Neutral, Angry, Sad, Happy, Fear, Surprise, and Disgust, covering both high-arousal and low-arousal emotional states.  
- **Audio Format:** 16 kHz single-channel `.wav` files, with 80-band mel-spectrograms extracted for model training and emotion embedding learning.  
- **Speaker Diversity:** Multiple male and female speakers spanning different age groups and vocal characteristics to prevent speaker bias and overfitting.  
- **Dataset Size:** 10–20 hours of annotated speech per language, enabling stable multilingual pretraining.  
- **Annotations:** Each utterance is labeled with emotion category, speaker identity, language tag, and normalized emotional intensity where available.

This database composition ensures balanced emotional coverage and sufficient acoustic diversity, forming a reliable foundation for learning disentangled and transferable emotion embeddings.


Full dataset details are available [here](https://github.com/NN-Project-2/Emotion-TTS-Emebddings/blob/main/README_1.md)


## 9. Zero-Shot Emotion Transfer and Control Scenarios in TTS  

- ✅ Emotion TTS  
- ✅ Cross-Lingual Transfer  
- ✅ Cross-Gender Emotion Transfer  
- ✅ Emotion Intensity Control  
- ✅ Integration with End-to-End TTS  


For more details, refer to the documentation. 🚀
