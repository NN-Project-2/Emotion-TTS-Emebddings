# EMOD: AN EFFICIENT APPROACH FOR LOW RESOURCE CONTROLLABLE EMOTIONAL SPEECH SYNTHESIS 🎤✨

## Overview
EMOD is a framework for enhancing expressive speech synthesis by capturing deep emotional embeddings from multilingual audio data. These embeddings integrate with end-to-end Text-to-Speech (TTS) models such as VITS and GPT-based speech models, enabling natural and controllable synthesis in low-resource language settings.

The extracted embeddings capture distinct emotions including happiness, sadness, anger, fear, surprise, and disgust, improving naturalness and human-likeness in generated speech. EMOD introduces fine-grained controllability, where deviations in pitch, energy, and duration are normalized into a continuous control factor. This parameter allows dynamic adjustment of emotional intensity, ranging from subtle expressivity to strong exaggeration.


### 🚀 **DEMO:** [Emotion-TTS Web](https://nn-project-2.github.io/Emotion-TTS-web/)
### 🎵 **Embeddings:** [Download Emotion embeddings.tar.xz](https://github.com/NN-Project-2/Emotion-TTS-Emebddings/blob/main/Emotion%20embeddings.tar.xz)

## 1. Table of Contents
1. [Overview](#overview)
2. [Introduction](#2-introduction)
    - [2.1 EMOD-Architecture](#21-emod-architecture)
    - [2.2 Intensity Control Parameter α](#22-intensity-control-parameter-α)
    - [2.3 Loss Functions (EMOD-Specific)](#23-loss-functions-emod-specific)
3. [Integration with End-to-End TTS](#3-integration-with-end-to-end-tts)
    - [3.1 VITS Architecture](#31-vits-architecture)
    - [3.2 GPT-Based TTS Architecture](#32-gpt-based-tts-architecture)
4. [Unsupervised Emotional Intensity Control](#4-unsupervised-emotional-intensity-control)
5. [Clustering for Emotion Cloning and Distance-Based Similarity](#6-clustering-for-emotion-cloning-and-distance-based-similarity)
6. [Test Setup and Results](#7-test-setup-and-results)
    - [7.1 Test Setup](#71-test-setup)
    - [7.2 Results Across Languages](#72-results-across-languages)
    - [7.3 Emotion Transfer and Speaker Cloning Results](#73-emotion-transfer-and-speaker-cloning-results)
7. [Emotional Embedding Database](#8-emotional-embedding-database)
8. [Zero-Shot Emotion Transfer and Control Scenarios in TTS](#9-zero-shot-emotion-transfer-and-control-scenarios-in-tts)


## 2. Introduction

EMOD is proposed to address the lack of controllable and transferable emotional representations in multilingual text-to-speech systems. Existing TTS models often entangle emotion with speaker identity and language-specific prosody, limiting zero-shot emotion transfer. EMOD explicitly learns language-independent emotional embeddings that can be reused across speakers, languages, and synthesis architectures. By separating emotional identity from intensity and speaker characteristics, EMOD enables fine-grained, continuous emotional control in low-resource and zero-shot settings. The extracted embeddings are directly compatible with both VITS and GPT-based TTS models without retraining the emotion encoder.



<p align="center">
  <img src="Architecture/1Emod.png" alt="EMOD Architecture">
</p>



## 2.1 EMOD-Architecture

EMOD generates emotional embeddings by explicitly disentangling emotional attributes from speaker identity and linguistic content, enabling **controllable emotion transfer without altering speaker timbre**. Emotional expressivity is modeled through a unified representation that integrates **prosodic, spectral, and phonetic descriptors**. Fundamental frequency ($F_0$), extracted using **PyWorld** and interpolated across unvoiced regions, captures pitch dynamics associated with emotional arousal, while **normalized short-term energy** reflects variations in vocal effort. **Phoneme-level duration** encodes temporal expressivity, capturing rhythm and speaking-rate changes correlated with emotional state. Spectral and linguistic content is represented using **80-dimensional mel-spectrograms** and **HuBERT embeddings**, where HuBERT provides phonetic abstraction while minimizing speaker and prosody bias.

These acoustic and linguistic cues are consolidated into a compact **192-dimensional feature representation** composed of mel coefficients, $F_0$, energy, and reduced HuBERT embeddings, ensuring robustness across speakers and languages. The fused features are processed by the **Emotion Encoder**, which combines **convolutional layers** for local temporal modeling, **bidirectional LSTMs** for sequence-level dependency capture, and **attention mechanisms** for dynamic weighting of emotionally salient regions. The encoder projects the input into a **256-dimensional latent emotion embedding** $z_{emo}$, which encodes categorical emotional identity independent of speaker characteristics.

In parallel, a **Speaker Encoder** extracts a **256-dimensional speaker embedding** $z_{spk}$ from reference speech, capturing speaker-specific timbre and vocal-tract characteristics while remaining invariant to emotional content. The emotion and speaker embeddings are then fused through concatenation to form a **512-dimensional joint latent representation**:

$$
z_{joint} = [\, z_{spk} \; \Vert \; z_{emo} \,]
$$

This fused representation provides the decoder with explicit access to both speaker identity and emotional attributes without requiring implicit disentanglement.

To generate acoustic outputs, EMOD employs a **cross-attention–based decoder** that conditions on the joint latent representation. Cross-attention allows the decoder to dynamically attend to emotion and speaker subspaces at each decoding step, enabling precise alignment between linguistic content, speaker timbre, and emotional prosody. This mechanism ensures that emotional modulation influences pitch, energy, and spectral coloration without corrupting phonetic structure or speaker identity.

Continuous emotional control is introduced at the latent level through an **intensity-controlled emotion embedding**. The global intensity parameter $\alpha$ modulates the emotion embedding before fusion:

$$
\tilde{z}_{emo} = \alpha \cdot (z_{emo} \odot r)
$$

where $r \in \mathbb{R}^{256}$ is a **dimension-wise modulation vector** that enables fine-grained control over specific emotional attributes. The scaled emotion embedding z̃_emo replaces z_emo in the fusion process, allowing emotion strength to be adjusted continuously while preserving categorical identity. This design enables EMOD to support **neutral-to-expressive interpolation**, **emotion exaggeration**, and **cross-lingual zero-shot synthesis** using a single fixed emotion space.



## 2.2 Intensity Control Parameter $\alpha$

While categorical emotion embeddings encode **emotional type**, they do not represent the **degree** to which an emotion is expressed. In natural speech, emotional strength varies continuously depending on context, intent, and prosodic realization. EMOD explicitly models this variation using a **continuous global intensity parameter** $\alpha \in [0, 1]$, which separates **emotional magnitude** from **emotional identity** and enables controlled modulation without altering the semantic direction of the emotion embedding.

The parameter $\alpha$ is derived from **low-level prosodic deviations** relative to speaker-specific neutral baselines, ensuring invariance to individual speaking styles. For each utterance, deviations in **pitch** $(\Delta F_0)$, **energy** $(\Delta E)$, and **duration** $(\Delta D)$ are computed with respect to neutral reference statistics of the same speaker. These deviations are linearly combined using learnable weights and normalized through a sigmoid function to obtain a bounded intensity estimate:

$$
\alpha = \sigma(\beta_1 \Delta F_0 + \beta_2 \Delta E + \beta_3 \Delta D)
$$

where $\sigma$ denotes the **sigmoid function** and $\beta_i$ are **trainable coefficients** that adaptively weight the contribution of each prosodic cue.

During inference, $\alpha$ acts as a **global scaling factor** applied directly to the emotion embedding:

$$
\tilde{z}_{emo} = \alpha \cdot z_{emo}
$$

This multiplicative interaction enables **smooth interpolation between neutral and expressive speech** while preserving the orientation of the emotion vector in latent space. The formulation remains compatible with **dimension-wise modulation vectors** $r \in \mathbb{R}^{256}$, enabling joint control over **global intensity** and **fine-grained emotional attributes** during synthesis.

## 2.3 EMOD Training Strategy

Training EMOD requires carefully balancing **emotion discriminability**, **speaker invariance**, and **cross-lingual generalization**, while maintaining full compatibility with downstream TTS models. To achieve this, EMOD is trained using a **three-stage pipeline** that progressively structures and stabilizes the emotion embedding space.

In the **first stage**, **multilingual pre-training** is performed using speech data from **Tamil, Malayalam, Hindi, English, Kannada, and Telugu**, supplemented with **Assamese, Marathi, and Punjabi**. This stage encourages the Emotion Encoder to learn **language-agnostic emotional cues** by exposing it to diverse phonetic inventories, intonation patterns, and prosodic realizations while enforcing a **shared unified emotion space** across languages.

The **second stage** focuses on **representation disentanglement**, where **orthogonality constraints** and **adversarial speaker classification** are applied to suppress speaker identity leakage into the emotion embeddings. This stage ensures that $z_{emo}$ encodes **only emotional information**, making the embeddings robust and transferable across **unseen speakers, genders, and languages** without compromising emotional consistency.

In the  EMOD embeddings are aligned with **TTS-specific optimization objectives** to ensure stable and interpretable conditioning during synthesis. While the **Emotion Encoder remains fixed**, downstream TTS components are optimized to correctly interpret and respond to emotional embeddings under varying **intensity levels**. All models are trained for **500,000 steps** using **AdamW** with a learning rate of $3 \times 10^{-4}$, a **batch size of 32**, **gradient accumulation over two steps**, and **mixed-precision training** on **four NVIDIA L4 GPUs** (24 GB VRAM each). The training corpus consists of approximately **10–20 hours of speech per language**, sampled at **16 kHz**.

## 2.3 Loss Functions (EMOD-Specific)

The EMOD framework optimizes a **fixed emotional embedding space** intended for reuse across multiple downstream TTS architectures. The loss design targets four explicit constraints: **(i) categorical emotion separability**, **(ii) speaker invariance**, **(iii) controllable emotional intensity**, and **(iv) acoustic realizability under latent conditioning**. All losses are applied during EMOD training and remain **independent of downstream TTS fine-tuning**.

---

### Mean Squared Error Loss ($L_{MSE}$)

Mean Squared Error loss is computed between **predicted** and **ground-truth mel-spectrograms** produced by the EMOD decoder conditioned on emotion and speaker embeddings. This loss constrains the emotion embedding $z_{emo}$ to encode acoustic variations that are **realizable in the mel-spectral domain**. During training, scaling $z_{emo}$ with the intensity parameter $\alpha$ directly affects **pitch variance**, **energy distribution**, and **temporal dynamics**.  
$L_{MSE}$ enforces that these variations remain **bounded and spectrally coherent**, preventing unstable latent trajectories that would degrade synthesis quality when EMOD embeddings are injected into **VITS** or **GPT-based** decoders.

<p align="center">
  <img src="loss/mse.png" alt="MSE Loss" width="200">
</p>

---

### Generalized End-to-End Speaker Loss ($L_{GE2E}$)

GE2E loss is applied to **speaker embeddings** extracted from emotionally conditioned speech. The objective minimizes **intra-speaker embedding variance** while maximizing **inter-speaker separation** under emotional modulation. This loss constrains EMOD training such that speaker embeddings remain **invariant to changes introduced by emotion embeddings and intensity scaling**. As a result, emotional conditioning modifies **prosodic structure** without altering **speaker-specific timbre characteristics**, enabling **zero-shot emotion transfer** across unseen speakers.

<p align="center">
  <img src="loss/ge2e.png" alt="GE2E Loss" width="200">
</p>

---

### Cross-Entropy Loss for Emotion Classification ($L_{CE}^{emo}$)

Cross-entropy loss is applied to an **auxiliary emotion classifier** attached to the emotion embedding $z_{emo}$. This loss enforces **linear separability of categorical emotions** in the latent space. During training, gradients from $L_{CE}^{emo}$ directly shape the **geometry of the emotion embedding distribution**, ensuring that embeddings corresponding to different emotion labels occupy **non-overlapping regions**. This constraint remains active during **embedding interpolation** and **intensity scaling**, maintaining categorical consistency under continuous control.

<p align="center">
  <img src="loss/CE.png" alt="Cross-Entropy Loss" width="200">
</p>

---

### Adversarial Cross-Entropy Loss for Speaker Suppression ($L_{CE}^{spk}$)

An **adversarial speaker classifier** is trained to predict speaker identity from the emotion embedding $z_{emo}$. The classifier minimizes speaker classification loss, while the **Emotion Encoder maximizes this loss via gradient reversal**. This objective removes speaker-dependent information from the emotion embedding space. The resulting embeddings encode **emotional attributes** while remaining **invariant to speaker identity**, enabling reuse across speakers without retraining.

---

### Orthogonality Loss ($L_{orth}$)

Orthogonality loss minimizes the **inner product** between emotion embeddings $z_{emo}$ and speaker embeddings $z_{spk}$. This constraint enforces **decorrelation** between emotion and speaker subspaces prior to fusion. During decoding, the concatenated **512-dimensional latent representation** $[\, z_{emo} ; z_{spk} \,]$ is processed using **cross-attention**. Orthogonality ensures that emotion-related modulation does not project onto speaker-dependent dimensions, preserving disentanglement under latent fusion.

<p align="center">
  <img src="loss/orth.png" alt="Orthogonality Loss" width="200">
</p>

---

### Intensity Consistency Loss ($L_{\alpha}$)

Intensity consistency loss constrains the predicted **global intensity parameter** $\alpha$ to match normalized prosodic deviations computed from **pitch** $(\Delta F_0)$, **energy** $(\Delta E)$, and **duration** $(\Delta D)$. The loss minimizes the discrepancy between predicted intensity and target intensity derived from **speaker-normalized prosodic statistics**. This enforces a **monotonic relationship** between $\alpha$ and acoustic expressivity, ensuring **predictable scaling of emotional strength** during inference.


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
