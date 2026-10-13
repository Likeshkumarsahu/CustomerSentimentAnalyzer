# Customer Sentiment Analyzer

A customer review sentiment analysis system powered by a fine-tuned DistilBERT model, with an interactive web interface built on Flask.

The application classifies customer text into three sentiment classes — **Negative**, **Neutral**, and **Positive** — and returns a confidence score for each prediction.

---

## Features

- Fine-tuned `distilbert-base-uncased` model for 3-class sentiment classification
- Clean, responsive web UI with live confidence visualization
- Lightweight Flask backend exposing a JSON inference API
- Reproducible pipeline: EDA, dataset balancing, training, domain evaluation, and explainability
- Token-level explanations via gradient-based attribution (Explainable AI)

---

## Model Performance

Evaluated on a held-out test set of 12,000 balanced samples:

| Metric | Score |
| --- | --- |
| Accuracy | 0.8627 |
| Weighted F1 | 0.8630 |
| Macro F1 | 0.86 |

Per-class F1: **Negative** 0.86 · **Neutral** 0.81 · **Positive** 0.92

---

## Tech Stack

- **Language:** Python 3.10+
- **Modeling:** Hugging Face Transformers, PyTorch
- **Backend:** Flask
- **Frontend:** HTML, CSS, Vanilla JavaScript
- **Data & Analysis:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn, WordCloud

---

## Project Structure

```
CustomerSentimentAnalyzer/
├── app.py                          # Flask web server and inference API
├── templates/
│   └── index.html                  # Web interface
├── sentiment_distilbert_best/      # Fine-tuned model (weights + tokenizer)
├── CSAnalyzer_EDA.ipynb            # Data exploration and preprocessing
├── CSAnalyzer_Model_Training.ipynb # DistilBERT fine-tuning
├── Domain_specific _sentiment.ipynb# Domain-wise performance evaluation
├── Explainable_AI.ipynb            # Token-level attribution
├── real_check.ipynb                # Real-world inference sanity check
├── Sentiment_Analysis_Model.ipynb  # Baseline LSTM model
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone <repository-url>
cd CustomerSentimentAnalyzer
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For CPU-only PyTorch installations, use the official index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Running the Application

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

Enter a review or any text into the input field and click **Analyze Sentiment**. The result displays the predicted sentiment and the model's confidence.

### API

**`POST /analyze`**

Request:

```json
{ "text": "The product works perfectly and arrived early." }
```

Response:

```json
{ "sentiment": "Positive", "confidence": 0.9987 }
```

| Field | Type | Description |
| --- | --- | --- |
| `sentiment` | string | `Negative`, `Neutral`, or `Positive` |
| `confidence` | float | Prediction confidence between 0 and 1 |

---

## Dataset

The model was trained on a balanced subset of the **Amazon Fine Food Reviews** dataset:

- **Source:** 568,454 raw reviews
- **Preprocessing:** HTML stripping, null removal, and text aggregation of title and body
- **Labeling:** rating 1–2 → Negative · 3 → Neutral · 4–5 → Positive
- **Balancing:** 40,000 samples per class → **120,000 total**
- **Split:** 80% train / 10% validation / 10% test (stratified)

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Base model | `distilbert-base-uncased` |
| Max sequence length | 256 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Optimizer | AdamW |
| Best-model metric | Weighted F1 |

The trained model is serialized to `sentiment_distilbert_best/` and loaded in read-only mode for inference.

---

## Notes and Limitations

- Designed for English-language customer reviews; performance on other domains or languages may vary.
- Neutral class is the most challenging to classify (F1 0.81) due to inherent ambiguity in mixed reviews.
- The model is intended for inference only and should not be used as the sole basis for business-critical decisions.
