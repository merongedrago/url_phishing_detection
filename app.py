import json
import os

# Ensure working directory is always the project root (where app.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from urllib.parse import urlparse

# ── RF feature functions (must be defined at module level so joblib can unpickle) ─
def get_url_string(df):
    return df["url"]


def safe_urlparse(url):
    try:
        return urlparse(url)
    except ValueError:
        return urlparse("http://default.com")


def extract_url_features(df):
    urls = df["url"]
    features = pd.DataFrame()
    features["url_length"] = urls.str.len()
    features["num_dots"] = urls.str.count(r"\.")
    features["num_hyphens"] = urls.str.count(r"-")
    features["num_slashes"] = urls.str.count(r"/")
    features["num_digits"] = urls.str.count(r"\d")
    features["num_special_chars"] = urls.str.count(r"[@_!#$%^&*()<>?|}{~:]")
    features["has_https"] = urls.str.startswith("https").astype(int)
    features["has_ip"] = urls.str.contains(
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    ).astype(int)
    features["has_at_symbol"] = urls.str.contains("@").astype(int)
    features["subdomain_count"] = urls.apply(
        lambda u: len(safe_urlparse(u).netloc.split(".")) - 2
    )
    features["path_length"] = urls.apply(lambda u: len(safe_urlparse(u).path))
    features["query_length"] = urls.apply(lambda u: len(safe_urlparse(u).query))
    return features


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🔒",
    layout="wide",
)

st.title("🔒 Phishing URL Detection")
st.caption(
    "Comparing a Random Forest baseline against a Character-Level BiLSTM "
    "on two independent datasets."
)

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_NAMES = ["benign", "phishing"]

# RF metrics hardcoded from analyis.ipynb outputs
RF = {
    "primary": {
        "accuracy": 0.9965,
        "precision": 0.9965,
        "recall": 0.9965,
        "f1": 0.9965,
        "roc_auc": None,
        "confusion_matrix": [[69033, 115], [239, 31610]],
        "fn_rate": 239 / 31849,
    },
    "kaggle": {
        "accuracy": 0.2028,
        "precision": 0.7901,
        "recall": 0.2028,
        "f1": 0.1034,
        "confusion_matrix": None,
        "pred_phishing_rate": 0.9733,
        "actual_phishing_rate": 0.1802,
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_datasets():
    phish_df = pd.read_csv("data/Phishing URLs.csv").rename(columns={"Type": "type"})
    phish_df["type"] = phish_df["type"].str.lower()
    real_df = pd.read_csv("data/URL dataset.csv")
    kaggle_df = pd.read_csv("data/kaggle_dataset.csv")

    prim_df = pd.concat([phish_df, real_df], ignore_index=True)
    prim_df.loc[prim_df["type"] == "legitimate", "type"] = "benign"
    prim_df = prim_df[["url", "type"]].dropna()
    return prim_df, kaggle_df


@st.cache_data
def load_bilstm_results():
    path = "output/bilstm_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: float, suffix: str = "", delta: str | None = None):
    st.metric(label=label, value=f"{value:.4f}{suffix}", delta=delta)


def confusion_heatmap(cm: list[list[int]], title: str) -> go.Figure:
    cm_arr = np.array(cm)
    fig = px.imshow(
        cm_arr,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=LABEL_NAMES,
        y=LABEL_NAMES,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=350)
    return fig


def bar_comparison(
    metrics: list[str], rf_vals: list[float], lstm_vals: list[float | None], title: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Random Forest", x=metrics, y=rf_vals, marker_color="#3B82F6")
    )
    if lstm_vals and any(v is not None for v in lstm_vals):
        fig.add_trace(
            go.Bar(
                name="BiLSTM",
                x=metrics,
                y=[v if v is not None else 0 for v in lstm_vals],
                marker_color="#F59E0B",
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        yaxis=dict(range=[0, 1.05], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=60, b=30),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
# ── Live inference helpers ─────────────────────────────────────────────────────
class PhishingBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out = self.dropout(F.relu(self.fc1(pooled)))
        return self.fc2(out).squeeze(-1)


@st.cache_resource
def load_bilstm_model():
    path = "output/models/bilstm_phishing.pt"
    if not os.path.exists(path):
        return None, None, None
    device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    hp = ckpt["hyperparams"]
    char2idx = ckpt["char2idx"]
    mdl = PhishingBiLSTM(
        vocab_size=hp["VOCAB_SIZE"],
        embed_dim=hp["EMBED_DIM"],
        hidden_dim=hp["HIDDEN_DIM"],
        num_layers=hp["NUM_LAYERS"],
        dropout=hp["DROPOUT"],
    ).to(device)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.eval()
    return mdl, char2idx, hp["MAX_LEN"]


@st.cache_resource
def load_rf_model():
    path = "output/models/rf_pipeline.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def bilstm_predict(url: str, mdl, char2idx, max_len: int) -> float:
    tokens = [char2idx.get(ch, 1) for ch in url[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    x = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logit = mdl(x).item()
    return float(torch.sigmoid(torch.tensor(logit)).item())


def rf_predict(url: str, pipeline) -> float:
    df = pd.DataFrame({"url": [url]})
    return float(pipeline.predict_proba(df)[0][1])


def gauge_chart(prob: float, title: str) -> go.Figure:
    color = (
        "#22c55e" if prob < 0.35
        else "#f97316" if prob < 0.65
        else "#ef4444"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 35], "color": "#dcfce7"},
                {"range": [35, 65], "color": "#ffedd5"},
                {"range": [65, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


tab1, tab2, tab3 = st.tabs(
    ["🔍 Live URL Check", "📊 Model Comparison", "📝 Discussion"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live URL Check
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter a URL to analyse")
    st.caption("Both models score the URL independently. Scores above 50% lean towards phishing.")

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/login?ref=abc123",
        label_visibility="collapsed",
    )

    run = st.button("Analyse", type="primary", use_container_width=False)

    if run and url_input.strip():
        bilstm_mdl, char2idx, max_len = load_bilstm_model()
        rf_pipeline = load_rf_model()

        rf_ready = rf_pipeline is not None
        bilstm_ready = bilstm_mdl is not None

        if not rf_ready:
            st.warning(
                "Random Forest model not found. "
                "Run all cells in `analysis/model1_rf.ipynb` to generate it."
            )
        if not bilstm_ready:
            st.warning(
                "BiLSTM model not found. "
                "Run all cells in `analysis/model2_bilstm.ipynb` through the Save model cell."
            )

        if rf_ready or bilstm_ready:
            col_rf, col_lstm = st.columns(2)

            with col_rf:
                if rf_ready:
                    with st.spinner("Running Random Forest…"):
                        rf_prob = rf_predict(url_input.strip(), rf_pipeline)
                    st.plotly_chart(
                        gauge_chart(rf_prob, "Random Forest"),
                        use_container_width=True,
                    )
                    verdict = (
                        "🟢 Likely safe" if rf_prob < 0.35
                        else "🟠 Suspicious" if rf_prob < 0.65
                        else "🔴 Likely phishing"
                    )
                    st.markdown(f"**{verdict}** ({rf_prob*100:.1f}% phishing probability)")
                else:
                    st.info("Random Forest unavailable")

            with col_lstm:
                if bilstm_ready:
                    with st.spinner("Running BiLSTM…"):
                        lstm_prob = bilstm_predict(url_input.strip(), bilstm_mdl, char2idx, max_len)
                    st.plotly_chart(
                        gauge_chart(lstm_prob, "BiLSTM"),
                        use_container_width=True,
                    )
                    verdict = (
                        "🟢 Likely safe" if lstm_prob < 0.35
                        else "🟠 Suspicious" if lstm_prob < 0.65
                        else "🔴 Likely phishing"
                    )
                    st.markdown(f"**{verdict}** ({lstm_prob*100:.1f}% phishing probability)")
                else:
                    st.info("BiLSTM unavailable")

            if rf_ready and bilstm_ready:
                st.divider()
                avg_prob = (rf_prob + lstm_prob) / 2
                if avg_prob < 0.35:
                    st.success(f"**Combined verdict: Likely safe** — average score {avg_prob*100:.1f}%")
                elif avg_prob < 0.65:
                    st.warning(f"**Combined verdict: Suspicious** — average score {avg_prob*100:.1f}%")
                else:
                    st.error(f"**Combined verdict: Likely phishing** — average score {avg_prob*100:.1f}%")

    elif run:
        st.info("Please enter a URL first.")

    with st.expander("How do the scores work?"):
        st.markdown(
            """
- **Random Forest**: uses TF-IDF character n-grams (3–5) plus 12 handcrafted URL statistics.
  Strong on in-distribution URLs; can overfire on benign URLs with unusual structure.
- **BiLSTM**: processes the URL character by character using a bidirectional LSTM.
  More robust to distribution shift since its vocabulary is just the character set.
- **Threshold**: 50% is the default decision boundary; scores in the 35–65% band are uncertain.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    bilstm = load_bilstm_results()

    def comparison_bar(primary_acc, primary_prec, primary_f1,
                       kaggle_acc, kaggle_prec, kaggle_f1, model_name):
        """Grouped bar chart: Accuracy, Precision & F1 — Primary vs Kaggle."""
        metrics = ["Accuracy", "Precision", "F1"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Primary Dataset",
            x=metrics,
            y=[primary_acc, primary_prec, primary_f1],
            marker_color="#3B82F6",
            text=[f"{v:.1%}" for v in [primary_acc, primary_prec, primary_f1]],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Kaggle (Secondary)",
            x=metrics,
            y=[kaggle_acc, kaggle_prec, kaggle_f1],
            marker_color="#EF4444",
            text=[f"{v:.1%}" for v in [kaggle_acc, kaggle_prec, kaggle_f1]],
            textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            title=model_name,
            yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(t=60, b=20, l=10, r=10),
        )
        return fig

    col_rf, col_lstm = st.columns(2)

    # ── Random Forest column ──────────────────────────────────────────────────
    with col_rf:
        st.subheader("Random Forest")
        st.caption("TF-IDF char n-grams + 12 handcrafted URL features")

        st.plotly_chart(
            comparison_bar(
                RF["primary"]["accuracy"],
                RF["primary"]["precision"],
                RF["primary"]["f1"],
                RF["kaggle"]["accuracy"],
                RF["kaggle"]["precision"],
                RF["kaggle"]["f1"],
                "Random Forest — Accuracy, Precision & F1",
            ),
            use_container_width=True,
        )

        st.markdown(
            """
**Why these results?**

Primary dataset:
- Trained on this distribution, so n-gram patterns are familiar
- Near-perfect performance as expected

Kaggle dataset:
- Fixed TF-IDF vocabulary does not transfer to new URL styles
- Flags **97% of URLs as phishing** when only 18% are
- Classic distribution shift failure
            """
        )

    # ── BiLSTM column ─────────────────────────────────────────────────────────
    with col_lstm:
        st.subheader("BiLSTM")
        st.caption("Character-level Bidirectional LSTM — end-to-end learned features")

        if bilstm is None:
            st.warning(
                "BiLSTM results not found. Run all cells in `analysis/model2_bilstm.ipynb` "
                "through the Save metrics cell to generate `output/bilstm_results.json`."
            )
        else:
            bp = bilstm["primary"]
            bk = bilstm["kaggle"]

            st.plotly_chart(
                comparison_bar(
                    bp["accuracy"],
                    bp["precision"],
                    bp["f1"],
                    bk["accuracy"],
                    bk["precision"],
                    bk["f1"],
                    "BiLSTM — Accuracy, Precision & F1",
                ),
                use_container_width=True,
            )

            st.markdown(
                """
**Why these results?**

Primary dataset:
- Learns character-level patterns end-to-end
- Competitive with Random Forest on familiar data

Kaggle dataset:
- Character set (a-z, digits, punctuation) is universal, so no vocabulary mismatch
- Reads URLs left-to-right and right-to-left to catch prefix and suffix cues
- Much smaller performance drop compared to Random Forest
                """
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Discussion
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Discussion")

    st.markdown(
        """
### Strengths & Weaknesses

| | Random Forest | BiLSTM |
|---|---|---|
| Primary accuracy | 99.65% | Competitive |
| Kaggle generalisation | Collapses | Holds up |
| Training speed | Seconds | Minutes |
| Interpretability | Feature importances | Black box |
| Feature engineering | 12 handcrafted features | None |
| Distribution shift risk | High | Low |

---

### Conclusions

- Both models score near-perfect on the primary dataset, but that overstates real-world performance
- The Random Forest fails on Kaggle due to vocabulary mismatch
- The BiLSTM holds up better because its character set is universal
- Missing a phishing URL is the costliest error, so false negative rate matters most

---

### Future Work

- **Late fusion**: combine BiLSTM embeddings with RF handcrafted features
- **Longer sequences**: URLs over 100 chars get truncated; a Transformer would handle these
- **Transfer learning**: a pre-trained model like URLTran or SecureBERT would likely outperform both
- **Adversarial testing**: evaluate against homoglyph attacks and subdomain abuse
        """
    )

    st.divider()
    st.caption("Project: Phishing URL Detection | Duke MIDS | ML for Cybersecurity")
