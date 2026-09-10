# LSTM v1 — Amazon Review Helpfulness Prediction
### Presentation Brief

> **Dataset-size note:** figures below are from a run on `amazon_reviews_s10.parquet` when it
> had 846,757 rows; the committed file now has 942,176. A fresh run on the current file has
> been verified and gives very similar results (F1 0.6272, AUC 0.7183) — see `06_lstm_analysis_report.md`
> and `README.md` for detail.

---

## 1. The Business Problem

Amazon reviews drive purchasing decisions, but **only a fraction are actually "helpful"** to shoppers. Surfacing helpful reviews faster improves:

- **Customer experience** — shoppers find trusted signals quickly.
- **Conversion & trust** — better reviews = higher purchase confidence.
- **Moderation efficiency** — reduces human effort in triaging millions of reviews.

**Question we answer:** *Can we automatically predict whether a review will be marked helpful — using both what it says and how it's written?*

---

## 2. Approach in One Picture

```
Raw Review  →  Clean Text  →  Tokenize (vocab=50k)  →  ┐
                                                       ├──► Hybrid BiLSTM  →  Helpful? (0/1)
Metadata (rating, length, popularity, verified, …)  ──►┘
```

Progressive model ladder (from simple to advanced) so every added complexity is **justified by measurable lift**.

---

## 3. Dataset Snapshot

| Item | Value |
|---|---|
| Source | `amazon_reviews_s10.parquet` |
| Total samples | **846,757** |
| Train / Test | 677,405 / 169,352 |
| Class balance | 52.5% not helpful / 47.5% helpful |
| Max sequence length | 200 tokens |
| Vocabulary | 50,000 (99.6% train coverage) |
| Metadata features | 13 (structural + behavioral) |

**Business takeaway:** balanced labels → standard metrics (F1, AUC) are trustworthy without heavy resampling.

---

## 4. Models Compared (Why Each One?)

| Model | Role | What it tests |
|---|---|---|
| **Logistic Regression (TF-IDF)** | Transparent baseline | Can simple word frequencies already solve this? |
| **Naive Bayes (TF-IDF)** | Probabilistic baseline | Are individual tokens strongly predictive? |
| **Random Forest (TF-IDF)** | Non-linear baseline | Do feature interactions matter? |
| **Vanilla LSTM** | Sequence model | Does word order add value? |
| **BiLSTM + GloVe** | Pretrained + bidirectional | Does external semantic knowledge help? |
| **Hybrid BiLSTM (Text + Metadata)** | Full fusion model | Does combining language + behavior win? |

This is an **ablation ladder**: each step answers a specific research question, not just a leaderboard entry.

---

## 5. Results — Headline Numbers

| Model | Accuracy | F1 | Precision | Recall | AUC |
|---|---:|---:|---:|---:|---:|
| **Hybrid BiLSTM** | **0.660** | **0.638** | **0.645** | **0.630** | **0.720** |
| BiLSTM + GloVe | 0.640 | 0.602 | 0.634 | 0.573 | 0.690 |
| Vanilla LSTM | 0.639 | 0.607 | 0.628 | 0.588 | 0.688 |
| Random Forest | 0.635 | 0.579 | 0.639 | 0.529 | 0.679 |
| Logistic Regression | 0.624 | 0.584 | 0.616 | 0.554 | 0.669 |
| Naive Bayes | 0.621 | 0.587 | 0.609 | 0.566 | 0.666 |

### Key takeaways
- **Hybrid BiLSTM wins across every metric.** Mixing *what reviewers say* with *how they review* gives the biggest lift.
- Deep models beat classical baselines by ~2–3 F1 points — meaningful at scale, not transformational.
- Pretrained GloVe did **not** outperform a plain LSTM → domain mismatch; embeddings trained on generic web text don't perfectly transfer to product reviews.

---

## 6. What Actually Drives "Helpful"? (SHAP on Metadata)

Top global drivers (mean |SHAP|):

| Rank | Feature | Business meaning |
|---|---|---|
| 1 | `review_length` | Longer reviews feel more substantive |
| 2 | `title_to_text_ratio` | Balanced, non-clickbait titles |
| 3 | `log_popularity` | Popular products attract scrutinized reviews |
| 4 | `days_since_first_review` | Timing / recency matters |
| 5 | `title_length` | Descriptive titles build trust |
| 6 | `images` | Visual evidence boosts credibility |
| 7 | `rating` | Star rating correlates with perceived usefulness |

**Insight:** helpfulness is **structural + behavioral**, not just linguistic. A review's *shape* matters as much as its *words*.

### LIME (local text insight)
For individual predictions, the model keys on topical and sentiment words (e.g., `app`, `video`, `like`, `offer`) — confirming it learns **plausible signals**, not random noise.

---

## 7. Technical Highlights

- **Leakage-safe pipeline** — vocabulary built from training set only.
- **Dual-branch architecture** — BiLSTM encodes text; an MLP processes normalized metadata; concatenated representation feeds a classifier head.
- **Early stopping** on BiLSTM + GloVe (epoch 5) — prevented overfitting when validation accuracy plateaued.
- **Explainability layer** — LIME for text-local reasoning, SHAP for global metadata attribution.
- **Tooling** — PyTorch, scikit-learn, SHAP, LIME, GloVe 6B.

---

## 8. Honest Limitations

- AUC of **0.72** → useful discrimination, but **not high-stakes automation grade**.
- Label is *operational* (`is_helpful` votes), which reflects platform exposure bias, not intrinsic quality.
- Some metadata features are sparse/low-variance in subsets.
- GloVe embeddings underperformed — suggests domain-specific embeddings or transformers would help.

---

## 9. Business Interpretation

| Use case | Recommended? | Why |
|---|---|---|
| **Ranking / sorting helpful reviews** | Yes | AUC 0.72 is strong for triage |
| **Highlighting top reviews on product pages** | Yes, with threshold tuning | Precision can be dialed up |
| **Fully automatic moderation gate** | **Not yet** | Needs higher precision + fairness audit |
| **Human-in-the-loop moderation assist** | Yes | Cuts reviewer workload meaningfully |

**Deployment stance:** ship as a **decision-support signal**, not a decision-maker.

---

## 10. Recommended Next Steps (Priority Order)

1. **Data quality audit** — feature variance, label consistency per shard.
2. **Threshold calibration** — tune for precision@k or cost-sensitive F1 per business need.
3. **Temporal validation** — test drift; does a model trained on older reviews still work?
4. **Stronger encoder** — benchmark a fine-tuned transformer (e.g., DistilBERT) for expected AUC lift.
5. **Explainability governance** — bootstrap SHAP confidence intervals, multi-sample LIME stability.

---

## 11. Executive Summary (30-Second Version)

> We built a hybrid deep-learning model that predicts whether an Amazon review will be considered helpful, combining the review's **language** (BiLSTM on text) with its **context** (metadata like length, popularity, rating). It outperforms all classical baselines with **AUC 0.72** and **F1 0.64**, and explainability confirms the drivers are intuitive: review length, title quality, product popularity, and imagery. The model is **production-ready as a ranking/triage signal** and should be paired with human review before any fully automated use. Next priority: **transformer-based encoder + threshold calibration**.

---

*Project: `lstm/lstmv1.ipynb` · Dataset: Amazon Reviews (s10 shard) · Stack: PyTorch, scikit-learn, SHAP, LIME*
