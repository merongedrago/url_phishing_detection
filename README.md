# Phishing URL Detection

A machine learning project comparing a **Random Forest** baseline against a **Character-Level Bidirectional LSTM** for detecting phishing URLs, with a focus on measuring generalisation under distribution shift.

**Live demo:** [urlphishingdetection.streamlit.app](https://urlphishingdetection.streamlit.app/) | **GitHub:** [merongedrago/url_phishing_detection](https://github.com/merongedrago/url_phishing_detection)

---

## Project Overview

Phishing URLs are a primary vector for credential theft and malware delivery. This project trains and evaluates two models on a combined dataset of ~500k URLs, then stress-tests both against an independent Kaggle holdout to measure how well each model generalises to unseen URL distributions.

The central research question: does a character-level neural model hold up better than a bag-of-words Random Forest when the URL distribution shifts?

**Datasets**
- **Primary:** [Mendeley Phishing URLs](https://data.mendeley.com/datasets/vfszbj9b36/1) + URL Dataset (~505k URLs) — used for training and evaluation
- **Holdout:** [Kaggle Malicious URLs](https://kaggle.com/datasets/sid321axn/malicious-urls-dataset) (~522k URLs, 18% phishing) — used only for out-of-distribution testing

---

## Models

### Model 1 — Random Forest
- **Features:** TF-IDF character n-grams (3–5) + 12 handcrafted URL statistics (length, dot count, hyphen count, digit count, presence of `@`, IP address patterns, HTTPS, subdomain depth, path/query length, etc.)
- **Primary accuracy:** 99.65%
- **Kaggle (OOD) accuracy:** 20.3% — flags 97% of URLs as phishing when only 18% are, a classic vocabulary mismatch failure

### Model 2 — LSTM
- **Architecture:** Embedding (vocab = character set) → Bidirectional LSTM (hidden dim 64, 1 layer) → mean pooling over non-padding tokens → FC(64) → sigmoid output
- **No feature engineering** — the model learns directly from the raw URL character sequence
- **Primary accuracy:** Competitive with Random Forest
- **Kaggle (OOD) accuracy:** Significantly more robust than Random Forest, since the character set (a–z, digits, punctuation) is universal across URL distributions

### Key Takeaway
The Random Forest collapses on out-of-distribution while The BiLSTM, operating on individual characters, generalises substantially better at the cost of slower training and reduced interpretability.

---

## Live Demo

The app is publicly deployed on Streamlit Community Cloud:
**[https://urlphishingdetection.streamlit.app/](https://urlphishingdetection.streamlit.app/)**

Enter any URL to receive independent phishing probability scores from both models, plus a combined verdict. Scores above 50% lean towards phishing; the 35–65% band is treated as uncertain.

---

## Running Locally (without Docker)

### Data Setup

The datasets are not included in this repository. Phishing URL datasets contain strings that trigger GitHub's secret scanning and push protection, so they cannot be pushed to GitHub.

To rerun the analysis notebooks, download the datasets and place them in a local `data/` folder at the project root:

```
final_cybersec_proj/
└── data/
    ├── Phishing URLs.csv       # https://data.mendeley.com/datasets/vfszbj9b36/1
    ├── URL dataset.csv         # https://data.mendeley.com/datasets/vfszbj9b36/1
    └── kaggle_dataset.csv      # https://kaggle.com/datasets/sid321axn/malicious-urls-dataset
```

> The `data/` folder is in `.gitignore` and will not be committed. The Streamlit app does not require the data files — only the pre-trained model files in `output/models/` are needed to run the app.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running with Docker

### 1. Build the image
```bash
docker build -t phishing-detector .
```

### 2. Run the container
```bash
docker run -p 8501:8501 phishing-detector
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Docker Desktop must be running. Download it at [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

---

## Project Structure

```
├── app.py                      # Streamlit dashboard
├── analysis/
│   ├── model1_rf.ipynb         # Model 1: Random Forest training & evaluation
│   └── model2_bilstm.ipynb     # Model 2: BiLSTM training & evaluation
├── output/
│   ├── bilstm_results.json     # Pre-computed BiLSTM metrics
│   ├── models/
│   │   ├── rf_pipeline.joblib  # Saved Random Forest pipeline
│   │   └── bilstm_phishing.pt  # Saved BiLSTM model weights
│   └── visualizations/         # Training curves, confusion matrices, ROC curves
├── Dockerfile
└── requirements.txt
```

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| **Live URL Check** | Enter any URL for real-time phishing probability scores from both models and a combined verdict |
| **Model Comparison** | Side-by-side accuracy, precision, and F1 on both the primary and Kaggle datasets; highlights the generalisation gap |

---

## Limitations & Future Work

- The Random Forest's TF-IDF vocabulary is frozen at training time, leading for any style not seen during training causes degraded performance
- The BiLSTM was trained for a limited number of epochs on the current dataset; training longer and on more diverse data would likely improve generalisation further
- URLs over ~100 characters get truncated by the BiLSTM; a Transformer architecture would handle long sequences better
- Possible future directions: late fusion of BiLSTM embeddings with handcrafted RF features, adversarial testing (homoglyph attacks, subdomain abuse), and training longer

---

## Acknowledgements

- [Claude](https://claude.ai) (Anthropic) was used to assist on creating LSTM model analysis, the interactive element of the streamlit website and generate this README. 
- Datasets sourced from Mendeley Data and Kaggle (see links above).

---

## References

- [Previous personal project for file structure](https://github.com/nogibjj/Meron_Gedrago_mini_Week12)
- [Previous personal project for streamlit and deployment](https://github.com/merongedrago/explainability_in_language)
- [Mendeley Phishing URL Dataset](https://data.mendeley.com/datasets/vfszbj9b36/1)
- [Kaggle Malicious URLs Dataset](https://kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org)
