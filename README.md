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
amazon-review/
├── data/
│   ├── processed/                       # sampled dataset, Git LFS
│   ├── sourced/                         # full corpus, git-ignored
│   ├── images/                          # review images, git-ignored
│   └── glove.6B.100d.txt                # GloVe embeddings, git-ignored
├── models/                              # saved checkpoints, git-ignored
├── notebook/                            # data + modelling pipeline
│   ├── 01_data_preparation.ipynb        # load, clean raw reviews
│   ├── 02_feature_engineering.ipynb     # engineer features
│   ├── 03_clustering_sampling.ipynb     # sample -> s03_filter.parquet
│   ├── 04_image_downloader.ipynb        # download images -> multimodal_s03_filter.parquet
│   ├── 05_visualisations.ipynb          # correlation + MI analysis
│   ├── 06_lstm.ipynb                    # LSTM / BiLSTM family
│   ├── 07_bert.ipynb                    # BERT + metadata fusion
│   ├── 08_multimodel.ipynb              # text + image + metadata fusion
│   ├── 09_sentiment_analysis.ipynb      # sentiment + emotion inference
│   └── train_test_split_protocol.ipynb  # split rationale, reference only
├── reports/
│   ├── data/                            # data reports
│   └── models/                          # model reports
├── environment.yml                      # conda spec
├── requirements.txt                     # pip requirements
└── FUTURE.md                            # known gaps
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

Run every notebook in `notebook/` in numeric order, 01 through 09:

```bash
jupyter nbconvert --to notebook --execute --inplace notebook/0<N>_*.ipynb
```

`train_test_split_protocol.ipynb` is reference-only and isn't part of the run order. Once
`03_clustering_sampling.ipynb` has produced `data/processed/s03_filter.parquet`, any of
06–09 can be run independently. `01_data_preparation.ipynb` downloads and cleans the full
43M-row raw dataset first, so budget time for that on a first run.

---

## Helpfulness prediction — model comparison

All three models are trained on the same `data/processed/s03_filter.parquet` /
`multimodal_s03_filter.parquet` (99,627 rows). Each notebook holds out its own test split
(different ratios, noted below), so test-set sizes differ; see the per-model reports in
`reports/models/` for detail.

| Model | Test set | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|---|
| **Hybrid BiLSTM** (best) | 19,926 (80/20 split) | 0.6708 | 0.6234 | 0.6023 | 0.6461 | 0.7267 |
| Multimodal (text + image + metadata) | 9,963 (80/10/10 split) | 0.6667 | 0.5870 | 0.6147 | 0.5616 | 0.7133 |
| BERT (DistilBERT + metadata fusion) | 14,945 (70/15/15 split) | 0.6551 | 0.5994 | 0.5876 | 0.6117 | — |

**Hybrid BiLSTM is the best model** — it leads on every reported metric (F1, AUC, and
accuracy), not just one.

---

## Sentiment & emotion analysis

Applied to the top-5 most-helpful reviews per product from `s03_filter.parquet` (n=10,000), using 
`cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment and
`j-hartmann/emotion-english-distilroberta-base` for emotion.

- Sentiment aligns well with star ratings at the extremes (1-star → negative, 5-star → positive)
- Mean top-1 confidence: 0.796, well above the 0.33 random baseline
- 4.7% conflict rate (469/10,000) between emotion and sentiment polarity — the two models are
  complementary, not interchangeable
- `fear` is unreliable (43.0% conflict rate vs. 1.4–17.2% for every other emotion) and should
  be excluded from downstream use

Sentiment and emotion answer different questions (polarity vs. affect), so there's no single
"best model" here — both are kept and reported together. LIME/SHAP walkthrough:
`notebook/09_sentiment_analysis.ipynb`, Step 4.

---

## Further reading

- `reports/data/` — data cards, feature engineering and sampling methodology
- `reports/models/` — per-model analysis reports
- `FUTURE.md` — known gaps and recommended next steps
