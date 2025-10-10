

### Key Database Highlights  

- **Languages Covered:**  
  Tamil, Malayalam, Hindi, English, Kannada, Telugu  
  (with supplementary data from Assamese, Marathi, and Punjabi for cross-lingual generalization).  

- **Emotion Categories:**  
  Neutral, Angry, Sad, Happy, Fear, Surprise, and Disgust, ensuring coverage of primary and secondary emotional states.  

- **Audio Quality:**  
  All recordings are normalized, down-sampled to **16 kHz**, stored in **.wav format**, and represented as **Mel spectrograms** for consistency across datasets.  

- **Speaker Diversity:**  
  Balanced representation of **male and female speakers** across different age groups and cultural backgrounds, minimizing speaker bias.  

- **Data Sources:**  
  Combination of open-source corpora (e.g., **RAVDESS, TORONTO-Emo, EmoDB**) and **custom recordings** from dramas, stories, and conversational speech to improve naturalistic expressivity.  

- **Annotation Protocols:**  
  - Each utterance is labeled with **categorical emotion ID, speaker ID, language, and gender**.  
  - **Intensity tags are derived in an unsupervised way**:  
    - Deviations in **pitch (∆F₀)**, **energy (∆E)**, and **phoneme duration (∆D)** are measured against each speaker’s neutral baseline.  
    - These deviations are normalized into a continuous scalar **α**, which is used during training and inference for controllable expressivity.  

- **Size and Balance:**  
  At least **10–20 hours per language**, with proportional distribution across emotional classes, preventing skew toward high-resource emotions or languages.  

- **Metadata:**  
  Includes **transcriptions**, **phoneme-level alignments**, and **prosodic feature statistics (F₀, energy, duration)** to support intensity estimation and embedding learning.  

---
