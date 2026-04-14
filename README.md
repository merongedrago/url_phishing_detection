# Phishing URL Detection

A machine learning project comparing three models — **Logistic Regression**, **Random Forest**, and a **Character-Level Bidirectional LSTM** — for detecting phishing URLs, with a focus on measuring generalisation under distribution shift.

**Live demo:** [urlphishingdetection.streamlit.app](https://urlphishingdetection.streamlit.app/) | **GitHub:** [merongedrago/url_phishing_detection](https://github.com/merongedrago/url_phishing_detection)

---

## Project Overview

Phishing URLs are a primary vector for credential theft and malware delivery. This project trains and evaluates three models on a combined dataset of ~500k URLs, then stress-tests all three against an independent Kaggle holdout to measure how well each generalises to unseen URL distributions.

The central research question: how do an interpretable linear model, a tree-based ensemble, and a character-level neural model compare under distribution shift?

**Datasets**
- **Primary:** [Mendeley Phishing URLs](https://data.mendeley.com/datasets/vfszbj9b36/1) + URL Dataset (~505k URLs) — used for training and evaluation
- **Holdout:** [Kaggle Malicious URLs](https://kaggle.com/datasets/sid321axn/malicious-urls-dataset) (~522k URLs, 18% phishing) — used only for out-of-distribution testing

---

## Models

### Model 1 — Logistic Regression
- **Features:** TF-IDF character n-grams (3–5) + 12 handcrafted URL statistics — same feature set as the Random Forest
- **Primary accuracy:** 75.5% — phishing recall 68.6%, FN rate 31.4%
- **Kaggle (OOD) accuracy:** 79.5% — generalises better than the Random Forest despite sharing the same features, as the linear decision boundary is less prone to overfitting the training vocabulary
- **Key advantage:** fully interpretable — coefficients directly show which n-grams and URL features drive phishing predictions

### Model 2 — Random Forest
- **Features:**  12 handcrafted URL statistics + TF-IDF character n-grams (3–5)
- **Primary accuracy:** 99.65%
- **Kaggle (OOD) accuracy:** 20.3% — flags 97% of URLs as phishing when only 18% are, a classic vocabulary mismatch failure

### Model 3 — BiLSTM
- **Architecture:** Embedding (vocab = character set) → Bidirectional LSTM (hidden dim 64, 1 layer) → mean pooling over non-padding tokens → FC(64) → sigmoid output
- **No feature engineering** — the model learns directly from the raw URL character sequence
- **Primary accuracy:** 97.8% — phishing recall 96.8%
- **Kaggle (OOD) accuracy:** 80.4% — most robust across datasets since its character-level vocabulary is universal

### Key Takeaway
The Random Forest achieves near-perfect accuracy on its training distribution but collapses under distribution shift. The Logistic Regression, despite sharing the same features, generalises better due to its simpler decision boundary. The BiLSTM achieves the best balance — strong primary performance and the smallest generalisation gap — at the cost of slower training and reduced interpretability.

---

## Live Demo

The app is publicly deployed on Streamlit Community Cloud:
**[https://urlphishingdetection.streamlit.app/](https://urlphishingdetection.streamlit.app/)**

Enter any URL to receive independent phishing probability scores from all three models, plus a combined verdict. Scores above 50% lean towards phishing; the 35–65% band is treated as uncertain.

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
│   ├── model2_bilstm.ipynb     # Model 2: BiLSTM training & evaluation
│   └── model3_logistic.ipynb   # Model 3: Logistic Regression training & evaluation
├── output/
│   ├── rf_results.json         # Pre-computed Random Forest metrics
│   ├── bilstm_results.json     # Pre-computed BiLSTM metrics
│   ├── lr_results.json         # Pre-computed Logistic Regression metrics
│   ├── models/
│   │   ├── rf_pipeline.joblib  # Saved Random Forest pipeline
│   │   ├── lr_pipeline.joblib  # Saved Logistic Regression pipeline
│   │   └── bilstm_phishing.pt  # Saved BiLSTM model weights
│   └── visualizations/         # Training curves, confusion matrices, ROC curves
├── Dockerfile
└── requirements.txt
```

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| **Live URL Check** | Enter any URL for real-time phishing probability scores from all three models and a combined verdict |
| **Model Comparison** | Side-by-side Recall & F1 for all three models on both the primary and Kaggle datasets; highlights the generalisation gap |

---

## Limitations & Future Work

- Both the Random Forest and Logistic Regression share a frozen TF-IDF vocabulary — any URL style unseen during training causes degraded performance
- The Logistic Regression's lower primary accuracy (75.5%) suggests the linear decision boundary struggles with the complexity of this feature space; further hyperparameter tuning (C, regularisation) may improve it
- The BiLSTM was trained for a limited number of epochs; training longer and on more diverse data would likely improve generalisation further
- URLs over ~100 characters get truncated by the BiLSTM; a Transformer architecture would handle long sequences better
- Possible future directions: late fusion of BiLSTM embeddings with handcrafted features, adversarial testing (homoglyph attacks, subdomain abuse), and pre-trained URL-specific models (URLTran, SecureBERT)

---

## Acknowledgements

- [Claude](https://claude.ai) (Anthropic) was used to assist on creating the LSTM and Logistic Regression model analysis, the interactive Streamlit dashboard, and this README. 
- Datasets sourced from Mendeley Data and Kaggle (see links above).

---

## References

- [Previous personal project for file structure](https://github.com/nogibjj/Meron_Gedrago_mini_Week12)
- [Previous personal project for streamlit and deployment](https://github.com/merongedrago/explainability_in_language)
- [Mendeley Phishing URL Dataset](https://data.mendeley.com/datasets/vfszbj9b36/1)
- [Kaggle Malicious URLs Dataset](https://kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org)
