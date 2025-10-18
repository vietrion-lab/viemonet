# Viemonet: A Hybrid Approach Integrating Contextualized Embeddings and Emotion Symbols for Sentiment Analysis

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Preserving affective signals through dual-branch architecture for Vietnamese sentiment analysis*

</div>

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Introduction](#-introduction)
- [Methodology](#-methodology)
  - [Overall Architecture](#overall-architecture)
  - [Input Representation](#input-representation)
  - [Dual-branch Encoders](#dual-branch-encoders)
  - [Fusion Layer](#fusion-layer)
- [Experimental Results](#-experimental-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Citation](#-citation)

---

## 🎯 Abstract

In this paper, we introduce a proposed model architecture that can capture the **affective signals** in Vietnamese sentiment analysis scenarios. The emotions are often removed in the preprocessing step, which can cause loss of the significant sentiment signal. To address this limitation, **Viemonet** is born at the intersection of linguistics and psychology.

This model splits the input into **two branches** and learns the final signals by aggregate layer. This approach shows **high performance** on two benchmark datasets, which has increased the accuracy of non-emotion symbols approaches by approximate **2%**.

**Key Contributions:**
- 🎭 A high-performance model that preserves emotion symbols during processing
- 🧠 A novel, psychologically-informed approach to sentiment analysis
- 🔄 Dual-branch architecture that models text-emotion interactions
- 📊 Rigorous evaluation demonstrating robustness on Vietnamese datasets

By operationalizing theories of emotional expression into a dual-branch architecture, we demonstrate that principles from psychology can directly inspire more effective and interpretable deep learning models.

---

## 🌟 Introduction

### Background

Sentiment Analysis underpins a wide range of user-understanding systems across social media, e-commerce, and customer support. In these settings, operational decisions—prioritizing tickets, monitoring brand crises, ranking products, routing agents—are highly sensitive to **affective signals** that are short, noisy, and rapidly evolving.

Recent advances in contextualized embeddings have substantially improved sentiment analysis by capturing contextual dependencies and enabling efficient domain adaptation. However, most contemporary pipelines remain predominantly text-only, and commonly used preprocessing steps inadvertently suppress or discard affective cues.

### The Power of Emotion Symbols

A major source of such cues is **emotion symbols**: emoji, emoticons, punctuation sequences, character elongation, and typographic emphasis. From linguistic and psychological perspectives, these symbols encode valence, arousal, and pragmatic intent. They help:

- 🎯 Disambiguate meaning
- 📈 Modulate intensity  
- 🔄 Invert polarity in sarcasm or irony

**Examples:**
```python
# Positive → Sarcastic
"Hay lắm" (English: "Very good") 😒  # Differs materially from:
"Hay lắm" (English: "Very good") 😊 🥳

# Pairing changes sentiment
"Tuyệt quá... :v" (English: "Excellent... :v") 
# ":v" often shifts from praise to mockery

# Removing emotions loses critical information  
"dm đéo hiểu sao suất 40k mà có mỗi miếng thịt :))))) 🥹"
# Without emoticons → ambiguous
# With :))))) and 🥹 → clearly frustrated/sarcastic
```

Removing or over-normalizing these signals eliminates information about position, frequency, repetition, and co-occurrence with words—precisely the patterns that reveal emphasis, indirect negation, or polarity flips.

### Challenges in Vietnamese

Modeling these phenomena is particularly challenging in Vietnamese:

- 📝 Syllable-segmented orthography
- 💬 Widespread colloquial/teencode forms
- 🔤 Frequent code-mixing and heavy use of symbols such as `"=))"`, `":v"`, or `"kkk"`
- 🎯 Stress tokenization schemes increase ambiguity near the Neutral boundary
- 📱 Short, fragmentary comments make a single emoji or punctuation sequence disproportionately influential

Public datasets also vary in emoji density, exhibit class imbalance and annotation noise, and often yield modest ceilings for models evaluated on emoji-rich domains.

---

## 🏗️ Methodology

### Overall Architecture

**Viemonet** employs a **dual-branch encoder architecture** that processes text and emotion symbols separately before fusing them through a meta-classifier:

```
Input Text: "Hay lắm :))))) 😊"
           ↓
    ┌──────┴──────┐
    ↓             ↓
[Comment]    [Emotions]
   Text      :))))) 😊
    ↓             ↓
┌─────────┐  ┌─────────┐
│ PhoBERT │  │ Emotion │
│  +CNN   │  │Knowledge│
│ Encoder │  │  Base   │
└────┬────┘  └────┬────┘
     │            │
  P_comment   P_emotion
     │            │
     └─────┬──────┘
           ↓
    ┌────────────┐
    │    Meta    │
    │ Classifier │
    └──────┬─────┘
           ↓
    Final Prediction
```

### Input Representation

The preprocessing pipeline is designed to intelligently handle Vietnamese social media text while preserving the critical affective signals encoded in emotion symbols. This process consists of three main stages: **emotion symbol extraction**, **emotion normalization**, and **text normalization**.

#### Emotion Symbol Extraction

The first stage separates plain text from emotion symbols (emojis and emoticons) using a sophisticated longest-match greedy algorithm. This approach is crucial because Vietnamese social media text often contains complex emoticon patterns where shorter emoticons can be substrings of longer ones (e.g., `:)` vs `:)))`).

The extraction process follows a two-phase strategy:

**Phase 1: Emoji Detection**
- Uses the `emoji` library to identify all Unicode emoji characters in the text
- Marks their positions in a character-level mask to prevent overlap with emoticon detection
- Preserves multi-codepoint emojis (e.g., combined emojis with skin tone modifiers)

**Phase 2: Emoticon Detection**
- Sorts the emoticon lexicon by length in descending order for longest-match priority
- Scans the text from left to right, attempting to match the longest possible emoticon at each position
- Handles repeated trailing characters (e.g., `:)))))` is recognized as an elongated version of `:)`)
- Marks matched positions to avoid double-counting

**Example:**
```
Input:  "ô hay kh hiểu sao suất 40k mà có mỗi miếng thịt :)))), hề thật ae ạ 🥹"

Output: 
  Comment:  "ô hay không hiểu sao suất 40 mà có mỗi miếng thịt , hề thật anh em ạ"
  Emotions: [':)))))', '🥹']
```

This separation is essential because it allows the model to:
1. **Process textual semantics independently** through the comment branch
2. **Capture affective signals explicitly** through the emotion branch
3. **Model their interactions** through the meta-classifier

Unlike traditional approaches that either remove emotions entirely or treat them as ordinary tokens, our method preserves both the positional information (where emotions appear) and their semantic content (what they express), enabling the model to learn sophisticated patterns like sarcasm detection (positive text + negative emotion) or emphasis amplification (multiple repeated emoticons).

---

## 📊 Experimental Results

### Benchmark Performance

Evaluation on three Vietnamese sentiment analysis datasets demonstrates **Viemonet's superiority**, especially on emoji-rich data:

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center"><b>Model</b></th>
      <th colspan="2" align="center"><b>UIT-VSMEC</b><br/>(With Emotions)</th>
      <th colspan="2" align="center"><b>UIT-VSMEC</b><br/>(Without Emotions)</th>
      <th colspan="2" align="center"><b>UIT-VSFC</b><br/>(Without Emotions)</th>
    </tr>
    <tr>
      <th align="center"><b>Accuracy</b></th>
      <th align="center"><b>F1-score</b></th>
      <th align="center"><b>Accuracy</b></th>
      <th align="center"><b>F1-score</b></th>
      <th align="center"><b>Accuracy</b></th>
      <th align="center"><b>F1-score</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><b>Viemonet</b></td>
      <td align="center"><b>76.62%</b></td>
      <td align="center"><b>74.33%</b></td>
      <td align="center">74.03%</td>
      <td align="center">72.29%</td>
      <td align="center">92.96%</td>
      <td align="center">82.05%</td>
    </tr>
    <tr>
      <td align="left">PhoBERT</td>
      <td align="center">73.33%</td>
      <td align="center">71.94%</td>
      <td align="center">73.45%</td>
      <td align="center">70.84%</td>
      <td align="center">92.96%</td>
      <td align="center">82.00%</td>
    </tr>
    <tr>
      <td align="left">VisoBERT</td>
      <td align="center">74.46%</td>
      <td align="center">72.72%</td>
      <td align="center">74.46%</td>
      <td align="center">71.93%</td>
      <td align="center">92.36%</td>
      <td align="center">80.40%</td>
    </tr>
    <tr>
      <td align="left">ViT5</td>
      <td align="center">74.60%</td>
      <td align="center">72.58%</td>
      <td align="center">74.46%</td>
      <td align="center">72.32%</td>
      <td align="center">92.29%</td>
      <td align="center">80.76%</td>
    </tr>
    <tr>
      <td align="left">VED_PhoBERT</td>
      <td align="center">75.76%</td>
      <td align="center">73.80%</td>
      <td align="center">73.88%</td>
      <td align="center">71.98%</td>
      <td align="center"><b>92.96%</b></td>
      <td align="center"><b>82.43%</b></td>
    </tr>
  </tbody>
</table>

</div>

### Key Findings

✅ **Superior Performance on Emotion-Rich Data (Primary Goal Achieved)**

Viemonet achieves **76.62% accuracy** and **74.33% F1-score** on UIT-VSMEC with emotions, outperforming all baseline models:
- **+3.29% accuracy improvement** over PhoBERT (73.33%)
- **+2.16% accuracy improvement** over VisoBERT (74.46%)
- **+2.02% accuracy improvement** over ViT5 (74.60%)
- **+0.86% accuracy improvement** over VED_PhoBERT (75.76%)

This demonstrates that Viemonet successfully captures and leverages emotion symbols for sentiment analysis in social media contexts where affective signals are abundant.

✅ **Emotion Symbols Are Significant Sentiment Signals**

The critical importance of emotion symbols is evidenced by the performance drop when emotions are removed from UIT-VSMEC:
- **Viemonet**: 76.62% → 74.03% (**-2.59% drop**)
- PhoBERT: 73.33% → 73.45% (+0.12%, marginal change but still lower than Viemonet)

This substantial drop in Viemonet's performance when emotions are absent confirms our hypothesis: **emotion symbols carry significant sentiment information that traditional text-only models cannot fully capture**. The fact that PhoBERT shows almost no change (slight increase) suggests it was already ignoring or under-utilizing emotion signals, while Viemonet's architecture is specifically designed to extract value from these signals.

Notably, even without emotion symbols (74.03%), Viemonet still outperforms standard PhoBERT (73.45%), indicating that the dual-branch training process helps the model learn more robust text representations.

✅ **Stable Performance on Non-Emotion Datasets**

On UIT-VSFC (formal product reviews with minimal emotion symbols), Viemonet achieves **92.96% accuracy** and **82.05% F1-score**, which is:
- **Competitive with PhoBERT** (92.96% accuracy, 82.00% F1)
- **Better than VisoBERT** (92.36% accuracy, 80.40% F1)
- **Better than ViT5** (92.29% accuracy, 80.76% F1)

This demonstrates that **Viemonet's architecture does not sacrifice performance on clean, formal text**. The model gracefully handles cases where emotion symbols are sparse or absent, showing that the dual-branch design is robust and generalizes well across different text domains. When emotions are not present, the emotion branch contributes neutral signals, and the meta-classifier effectively relies on the comment branch.

✅ **Robust Across Datasets with Varying Emotion Density**

Viemonet maintains consistent high performance across three datasets with different characteristics:
- **High emotion density** (UIT-VSMEC): Best performance (76.62%)
- **Low emotion density** (UIT-VSFC): Competitive performance (92.96%)

This versatility makes Viemonet suitable for real-world applications where text quality and emotion usage vary widely across users, platforms, and contexts.

### Dataset Characteristics

| Dataset | Description | Text Type | Emotion Density |
|:--------|:------------|:----------|:----------------|
| **UIT-VSMEC** | Social media comments | Informal, emoji-rich | High |
| **UIT-VSFC** | Product reviews | Formal, clean text | 0% |
---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM (16GB+ recommended)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/kaomoji-intergated-sentiment-analysis.git
cd kaomoji-intergated-sentiment-analysis
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Dependencies

```
# Core Libraries
datasets>=2.14.0
pandas>=1.5.0
PyYAML>=6.0

# Deep Learning & NLP
transformers>=4.30.0
torch>=2.0.0
peft>=0.4.0
accelerate>=0.20.0
evaluate>=0.4.0

# Vietnamese NLP
underthesea>=1.3.0

# Emotion Processing
emoji>=2.0.0

# Evaluation
scikit-learn>=1.3.0
```

---

## ⚡ Quick Start

### Basic Usage

```python
from viemonet import Trainer
from utils import load_dataset, load_config

# Load configuration
config = load_config("config.yaml")

# Load dataset from HuggingFace
dataset = load_dataset(config.datasets.social_comments.path)

# Initialize trainer with emotion-aware method
trainer = Trainer(
    method='separate_emotion',  # Use dual-branch architecture
    model_name='viemonet_phobert',
    dataset_name='uit-vsmec'
)

# Train the model
trainer.train(raw_data=dataset)
```

### Training Options

**Method Selection:**

```python
# 1. Viemonet: Dual-branch with emotion symbols (RECOMMENDED)
trainer = Trainer(method='separate_emotion', model_name='viemonet_phobert')

# 2. No Emotion: Remove emotions during preprocessing (baseline)
trainer = Trainer(method='no_emotion', model_name='phobert')

# 3. Text-only models for comparison
trainer = Trainer(method='no_emotion', model_name='visobert')
trainer = Trainer(method='no_emotion', model_name='vit5')
```

**Foundation Models:**

- `phobert`: PhoBERT (RoBERTa for Vietnamese)
- `visobert`: ViSoBERT (Sentence-BERT for Vietnamese)
- `vit5`: ViT5 (T5 for Vietnamese)

**Dataset Selection:**

```python
# From config.yaml
datasets:
  social_comments:
    path: viethq1906/UIT-VSMEC-Sentiment-Relabelled
    text_field: sentence
    label_field: sentiment
    
  vietnamese_comments:
    path: ura-hcmut/UIT-VSFC
    text_field: text
    label_field: label
```

### Inference Example

```python
from viemonet.preprocess.preprocess_pipeline import PreprocessPipeline
from viemonet.models.main_model_manager import MainModelManager

# Initialize preprocessing pipeline
pipeline = PreprocessPipeline(
    method='separate_emotion',
    foundation_model_name='phobert'
)

# Sample comments
comments = [
    "Hay lắm :))))) 😊",
    "Tệ quá đi mất :(",
    "Bình thường thôi"
]

# Preprocess
processed = pipeline(comments)

# Load trained model
model_manager = MainModelManager()
model = model_manager.load_model('viemonet_phobert')

# Predict
predictions = model.predict(
    input_ids=processed['input_ids'],
    attention_mask=processed['attention_mask'],
    emotions=processed['emotions']
)

# Results: ['positive', 'negative', 'neutral']
```

---

## 📁 Project Structure

```
kaomoji-intergated-sentiment-analysis/
├── config.yaml                      # Main configuration file
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── research/
│   ├── main.py                     # Training entry point
│   ├── utils.py                    # Utility functions
│   ├── config_schema.py            # Configuration schema
│   │
│   ├── viemonet/                   # Core package
│   │   ├── __init__.py
│   │   │
│   │   ├── config/                 # Configuration management
│   │   │   ├── __init__.py
│   │   │   ├── env.py
│   │   │   └── train_config.py
│   │   │
│   │   ├── constant/               # Constants and enums
│   │   │   ├── config_constant.py
│   │   │   └── training_constant.py
│   │   │
│   │   ├── models/                 # Model architectures
│   │   │   ├── __init__.py
│   │   │   ├── emotion_sentiment.py        # Emotion knowledge base
│   │   │   ├── foundation_model_manager.py # Pre-trained models
│   │   │   ├── head_model_manager.py       # Classification heads
│   │   │   └── main_model_manager.py       # Model factory
│   │   │   │
│   │   │   ├── foundation_models/          # Base encoders
│   │   │   │   ├── phobert_lora.py
│   │   │   │   ├── visobert_lora.py
│   │   │   │   └── vit5_lora.py
│   │   │   │
│   │   │   ├── cls_head/                   # Classification heads
│   │   │   │   ├── base_head.py
│   │   │   │   ├── cnn.py
│   │   │   │   ├── gru.py
│   │   │   │   └── logreg.py
│   │   │   │
│   │   │   ├── main_models/                # Complete models
│   │   │   │   ├── viemonet_phobert.py    # ⭐ Main Viemonet model
│   │   │   │   ├── viemonet_visobert.py
│   │   │   │   ├── phobert.py
│   │   │   │   ├── visobert.py
│   │   │   │   └── vit5.py
│   │   │   │
│   │   │   └── submodels/                  # Model components
│   │   │       ├── comment_classifier.py   # Text branch
│   │   │       ├── emotion_classifier.py   # Emotion branch
│   │   │       ├── meta_classifier.py      # Fusion layer
│   │   │       └── mean_pool.py
│   │   │
│   │   ├── preprocess/             # Data preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── preprocess_pipeline.py     # Main pipeline
│   │   │   ├── preprocessor.py            # Data preparation
│   │   │   ├── emotion_dataset.py         # Dataset wrapper
│   │   │   └── vietnamese_abbrev.csv      # Abbreviation dict
│   │   │
│   │   ├── training/               # Training utilities
│   │   │   ├── trainer.py                 # Main trainer
│   │   │   ├── training_builder.py        # Training pipeline
│   │   │   ├── callbacks.py
│   │   │   └── data_collator.py
│   │   │
│   │   └── schemas/                # Data schemas
│   │       ├── config.py
│   │       └── emotion.py
│   │
│   └── experiment/                 # Experimental results
│       └── outputs_final/          # Final model outputs
│           ├── uit-vsmec/
│           ├── uit-vsfc/
│           └── uit-vsmec-ved/
│
└── venv/                           # Virtual environment
```

### Key Components

| Component | Description |
|:----------|:------------|
| **viemonet/models/main_models/viemonet_phobert.py** | Complete Viemonet architecture with dual-branch encoders |
| **viemonet/models/submodels/** | Individual components: comment classifier, emotion classifier, meta-classifier |
| **viemonet/models/emotion_sentiment.py** | Unified emotion knowledge base (emoticons + emojis) |
| **viemonet/preprocess/preprocess_pipeline.py** | Text preprocessing, emotion extraction, normalization |
| **viemonet/training/trainer.py** | Training orchestration with multi-task learning |
| **config.yaml** | Central configuration for datasets, models, and training |

---

## ⚙️ Configuration

### config.yaml Structure

```yaml
# Dataset Configuration
datasets:
  social_comments:
    path: viethq1906/UIT-VSMEC-Sentiment-Relabelled
    text_field: sentence
    label_field: sentiment
    name: Social Comments Dataset
    description: A dataset of social media comments for sentiment analysis.
    
  vietnamese_comments:
    path: ura-hcmut/UIT-VSFC
    text_field: text
    label_field: label
    name: Vietnamese Comments Dataset
    description: A dataset of Vietnamese comments for sentiment analysis.

# Training Settings
training_setting:
  max_length: 128                    # Maximum sequence length
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 500
  weight_decay: 0.01
  
  multi_task_learning:
    alpha: 0.3                       # Weight for comment classifier loss
    beta: 0.7                        # Weight for meta-classifier loss
    
  class_weights: [1.2, 1.0, 1.5]   # For handling class imbalance

# Emotion Knowledge Base
emotion_knowledge_base:
  emoticon_sentiment: data/emoticon_sentiment.csv
  emoji_sentiment: data/emoji_sentiment.csv

# Model Configuration
foundation_models:
  phobert:
    model_name: vinai/phobert-base
    use_lora: true
    lora_r: 16
    lora_alpha: 32
    
  visobert:
    model_name: uitnlp/visobert
    use_lora: true
    
  vit5:
    model_name: VietAI/vit5-base
    use_lora: true
```

### Customization

**Adjust Multi-Task Learning Weights:**

```yaml
training_setting:
  multi_task_learning:
    alpha: 0.3  # Comment classifier (lower = less influence)
    beta: 0.7   # Meta-classifier (higher = more influence)
```

**Experiment with Different Architectures:**

```python
# Try different head architectures
trainer = Trainer(
    method='separate_emotion',
    model_name='viemonet_phobert',
    head_name='cnn'  # Options: 'cnn', 'gru', 'logreg'
)
```

**Enable LoRA Fine-tuning:**

```yaml
foundation_models:
  phobert:
    use_lora: true
    lora_r: 16        # Rank of LoRA matrices
    lora_alpha: 32    # Scaling factor
    lora_dropout: 0.1
```

---

## 🎓 Citation

If you find this work useful, please cite:

```bibtex
@article{viemonet2025,
  title={Viemonet: A Hybrid Approach Integrating Contextualized Embeddings and Emotion Symbols for Sentiment Analysis},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaboration:

- **Email**: viethq.1906@gmail.com
- **GitHub**: [@hqvjet](https://github.com/hqvjet)

---

## 🙏 Acknowledgments

- **HuggingFace** for providing the Transformers library
- **UIT-NLP** for the Vietnamese sentiment analysis datasets
- **VinAI** for PhoBERT pre-trained model
- All contributors and researchers in Vietnamese NLP

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ and :) by the research team

</div>
