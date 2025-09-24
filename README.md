# 🎭 Vietnamese Emoticon/Kaomoji Sentiment Analysis

A Vietnamese sentiment analysis project focused on emoticons and kaomoji expressions in social media text.

## 📊 Dataset

- **File**: `data/vietnamese_emoticon_kaomoji_training_dataset.csv`
- **Size**: 1,589 samples with sentiment labels
- **Language**: Vietnamese
- **Content**: Social media comments containing emoticons/kaomoji
- **Labels**: `positive`, `negative`, `neutral`

### Dataset Distribution
- **Positive**: 926 samples (58.3%)
- **Neutral**: 422 samples (26.6%) 
- **Negative**: 241 samples (15.2%)

## 🚀 Quick Start

1. **Setup environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. **Configure HuggingFace token** (if needed):
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

3. **Run main script**:
```bash
cd research
python main.py
```

## 📁 Project Structure

```
├── data/
│   └── vietnamese_emoticon_kaomoji_training_dataset.csv  # Main training dataset
├── research/
│   ├── main.py              # Main processing script
│   ├── utils.py             # Utility functions
│   ├── schemas.py           # Data schemas
│   └── transform_dataset.py # Dataset transformation
├── config.yaml              # Configuration file
├── DATASET_REPORT.md        # Detailed dataset report
└── README.md               # This file
```

## 🎯 Features

- **Real Vietnamese Data**: 100% authentic social media content, no synthetic data
- **Emoticon Detection**: Supports Western emoticons (:), :D) and Vietnamese style (:)), :))))
- **Kaomoji Support**: Japanese-style emoticons (ಠ_ಠ, ◕_◕)
- **Multi-source**: Collected from 7 different Vietnamese datasets

## 📝 Usage Examples

The dataset contains Vietnamese social media comments with emoticons:

```csv
content,sentiment_label
"Mát thế này cứ như mùa đông giữa hè luôn :)))",positive
"Ve chắc hoang mang lắm, hè đâu mà kêu :)))",negative  
"Ai ngờ hè mà lôi chăn bông ra đắp :))",neutral
```

## 🤝 Contributing

This dataset is suitable for:
- Training Vietnamese sentiment analysis models
- Research on emoticon usage in Vietnamese culture
- Developing Vietnamese social media chatbots
- Gaming/forum community sentiment analysis

## 📄 License

See dataset sources in `DATASET_REPORT.md` for individual license information.

---

*Generated with ❤️ for Vietnamese NLP research*
