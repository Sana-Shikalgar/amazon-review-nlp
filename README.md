# Amazon Review

End-to-end pipeline for two tasks on the Amazon Electronics Reviews dataset
(McAuley-Lab/Amazon-Reviews-2023, via HuggingFace):

1. **Helpfulness prediction** — binary classification of `helpful_binary` (1 if
   `helpful_vote > 0`, else 0), compared across three models: an LSTM/BiLSTM family, a
   BERT-based fusion model, and a multimodal (text + image) model.
2. **Sentiment & emotion analysis** — inference-only pipeline applying two pre-trained
   HuggingFace models to the same review corpus, with LIME/SHAP explainability.

---

## Repository structure

```
data/
  processed/            — cleaned/sampled datasets (amazon_reviews_s10.parquet, s30.parquet), Git LFS
  multimodel/            — multimodal-specific dataset (dataset_50k.parquet), Git LFS
  splits/                — temporal train/val/test split exported by 05_visualization.ipynb, Git LFS
notebook/                — data pipeline, run in order
  01_data_preparation.ipynb    — load raw reviews, clean, filter
  02_feature_engineering.ipynb — engineer review_length, image_bucket, popularity, etc.
  03_clustering_sampling.ipynb — cluster-stratified sampling to amazon_reviews_s10.parquet
  04_image_downloader.ipynb    — download review images for the multimodal dataset
  05_visualization.ipynb       — EDA, correlations, mutual information, temporal train/val/test split
models/                  — model notebooks, numbered by progression in complexity
  01_lstm.ipynb           — TF-IDF baselines + Vanilla LSTM + BiLSTM+GloVe + Hybrid BiLSTM
  02_bert.ipynb            — DistilBERT + metadata fusion model
  03_multimodel.ipynb      — text + image + metadata fusion model
  sentiment_analysis.ipynb — sentiment + emotion inference and explainability (separate task, not numbered)
reports/
  data/                   — data cards, feature/sampling methodology reports
  models/                 — per-model analysis reports (LSTM overview, results, presentation)
environment.yml           — conda environment spec
requirements.txt          — pip requirements (see note on installing the CUDA torch build)
FUTURE.md                 — known gaps and recommended next steps
```

---

## Setup

### Conda

```bash
conda env create -f environment.yml
conda activate amazon-review
```

### Pip

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or your CUDA version
pip install -r requirements.txt
```

### Data

The processed datasets are tracked with Git LFS:

```bash
git lfs install
git lfs pull
```

---

## How to run

Run the data pipeline first, then any model notebook (each model notebook is independent
of the others once the data pipeline has produced `data/processed/amazon_reviews_s10.parquet`):

```bash
jupyter nbconvert --to notebook --execute --inplace notebook/01_data_preparation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebook/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebook/03_clustering_sampling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebook/04_image_downloader.ipynb   # needed for models/03_multimodel.ipynb
jupyter nbconvert --to notebook --execute --inplace notebook/05_visualization.ipynb

jupyter nbconvert --to notebook --execute --inplace models/01_lstm.ipynb
jupyter nbconvert --to notebook --execute --inplace models/02_bert.ipynb
jupyter nbconvert --to notebook --execute --inplace models/03_multimodel.ipynb            # ~4h/epoch — run in the background
jupyter nbconvert --to notebook --execute --inplace models/sentiment_analysis.ipynb
```

`01_data_preparation.ipynb` re-downloads and cleans the full 43M-row raw dataset unless
`data/processed/amazon_reviews_s30.parquet` already exists, in which case it verifies the
pipeline against a small streamed sample instead — see the notebook for details.

---

## Helpfulness prediction — model comparison

Fresh results from a full end-to-end run on this repo's committed data; see the per-model
reports in `reports/models/` for the underlying detail. Sample sizes differ across models by
design (LSTM uses the full processed dataset, BERT and the multimodal model use their own
independently-sized samples) — see `FUTURE.md` for the plan to make this genuinely
apples-to-apples.

| Model | Sample size | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|---|
| Hybrid BiLSTM (best of the LSTM family) | 942,176 (753,740 train / 188,436 test) | 0.6601 | 0.6272 | 0.6456 | 0.6098 | 0.7183 |
| BERT (DistilBERT + metadata fusion) | 48,705 (34,093 train / 7,306 val / 7,306 test) | 0.6755 | 0.5739 | 0.6149 | 0.5381 | — |
| Multimodal (text + image + metadata) | ~83,000 (with downloaded images) | _pending_ | _pending_ | _pending_ | _pending_ | — |

**Best model (so far): Hybrid BiLSTM** — leads on every metric that's comparable across
models (F1 0.6272, AUC 0.7183), on the largest of the three sample sizes. The multimodal
row is still training (see `FUTURE.md`); this section will be updated once it completes.
BERT's lower F1 despite a similar accuracy comes from a much smaller, differently-sourced
sample (48.7K reservoir-sampled reviews vs. the LSTM's full 942K-row processed dataset) —
not a like-for-like comparison. AUC is left blank for BERT since its notebook reports only
accuracy/precision/recall/F1, not class probabilities suitable for ROC-AUC.

---

## Sentiment & emotion analysis

Inference-only pipeline (`models/sentiment_analysis.ipynb`) — no training performed:

- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (positive / neutral / negative)
- **Emotion:** `j-hartmann/emotion-english-distilroberta-base` (joy, anger, sadness, fear, disgust, surprise, neutral)

**Pipeline steps:**

1. Load `data/processed/amazon_reviews_s10.parquet` (942,176 reviews, 12 columns) — title is
   already merged into `review_text` upstream, so `combined_text` here is just an alias of it
2. Filter to the top-5 most helpful reviews per product, ranked by `helpful_vote` with
   `review_length` as tiebreaker (942,176 → 599,361 reviews)
3. Run batched inference on `combined_text` with 512-token
   truncation
4. Store full class probability distributions alongside top-1 labels

**Key findings** (see `models/sentiment_analysis.ipynb` and `FUTURE.md` for full detail):

- Sentiment aligns well with star ratings at the extremes (1-star → negative, 5-star → positive)
- Mean top-1 confidence ≈ 0.80, well above the 0.33 random baseline for 3 classes
- The two models are complementary, not interchangeable — a 4.7% conflict rate exists between
  emotion and sentiment polarity (e.g. negative emotion paired with positive sentiment)
- The `fear` emotion label is unreliable (bimodal across ratings 1 and 5) and should be
  excluded from downstream use

Sentiment and emotion answer different questions (polarity vs. affect), so this task has no
single "best model" the way helpfulness prediction does — both models are kept and reported
together.

### Explainability

Both the sentiment and emotion models are explained with LIME (`lime.lime_text.LimeTextExplainer`,
200-perturbation local linear approximation) and SHAP (`shap.Explainer` with `shap.maskers.Text`,
Shapley-value token attribution) on representative examples per sentiment label. See
`models/sentiment_analysis.ipynb`, Step 4, for the full walkthrough.

---

## Further reading

- `reports/data/` — data cards, feature engineering and sampling methodology
- `reports/models/` — per-model analysis reports
- `FUTURE.md` — known gaps and recommended next steps
