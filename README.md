# 🔍 Fake News Detector

A machine learning project that analyzes news articles to detect potential misinformation using NLP techniques including text classification, sentiment analysis, and data visualization.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

## 🌟 Overview

This project builds a fake news classification pipeline that:

- **Cleans and preprocesses** raw news article text (URL removal, stopwords, lemmatization)
- **Converts text to numerical features** using TF-IDF (Term Frequency–Inverse Document Frequency)
- **Trains and compares multiple ML models** — Logistic Regression, Naive Bayes, and SMOTE-enhanced classifiers
- **Evaluates model performance** with precision, recall, F1-score, and confusion matrices
- **Analyzes sentiment bias** using VADER to detect emotionally charged language
- **Visualizes patterns** with word clouds, distribution charts, and feature importance plots

## 📊 Dataset

**Fake and Real News Dataset** — ~44,898 labeled news articles

| Class | Count |
|-------|-------|
| Fake News | ~23,481 |
| Real News | ~21,417 |

Source: [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## 🧪 Models & Results

| Model | Description |
|-------|-------------|
| **Logistic Regression** | TF-IDF features + balanced class weights |
| **Multinomial Naive Bayes** | Probabilistic classifier with Laplace smoothing |
| **Logistic Regression + SMOTE** | Synthetic oversampling for class balance |

## 🛠️ Technologies

- **pandas / NumPy** — Data manipulation
- **NLTK** — Text preprocessing (stopwords, lemmatization)
- **scikit-learn** — TF-IDF vectorization, model training, evaluation metrics
- **imbalanced-learn** — SMOTE for handling class imbalance
- **VADER Sentiment** — Sentiment and bias analysis
- **matplotlib / seaborn** — Visualization
- **WordCloud** — Word frequency visualization

## 📁 Project Structure

```
Fake-News-Detector/
├── Fake-News-Detector.ipynb   # Complete training notebook
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or Google Colab

### Run Locally

```bash
# Clone the repo
git clone https://github.com/isthatpaul/Fake-News-Detector.git
cd Fake-News-Detector

# Install dependencies
pip install pandas numpy scikit-learn nltk vaderSentiment wordcloud imbalanced-learn matplotlib seaborn

# Launch the notebook
jupyter notebook Fake-News-Detector.ipynb
```

### Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub**
3. Paste: `https://github.com/isthatpaul/Fake-News-Detector`
4. Select `Fake-News-Detector.ipynb`
5. Upload `Fake.csv` and `True.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
6. Run all cells in order

## 🔍 How It Works

```
Raw Article Text
       │
       ▼
┌─────────────────┐
│  Preprocessing   │  Remove URLs, mentions, HTML, special chars
│                  │  Lowercase → Remove stopwords → Lemmatize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    TF-IDF        │  Convert text → 10,000-feature numerical vectors
│  Vectorization   │  Unigrams + bigrams
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Classifier  │  Logistic Regression / Naive Bayes / LR + SMOTE
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction     │  REAL ✅ or FAKE ❌ + confidence score
│   + Sentiment    │  Bias level: 🟢 LOW / 🟡 MODERATE / 🔴 HIGH
└─────────────────┘
```

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) for the Fake and Real News Dataset
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for social media sentiment analysis
- [NLTK](https://www.nltk.org/) for natural language processing tools
