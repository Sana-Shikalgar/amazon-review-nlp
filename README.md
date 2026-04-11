# WM9B7-AIDL-AmazonReview

MSc Applied Artificial Intelligence — AI and Deep Learning (WM9B7)  
WMG, University of Warwick — 2025/26  
Client: IntelliSys Ltd.

---

## Project Overview

Binary classification task: predict whether an Amazon Electronics review will receive at least one helpful vote from other customers.

**Dataset:** 50,000 Amazon Electronics reviews (HuggingFace, sampled from `amazon_reviews_us_Electronics_v1_00`)  
**Target:** `helpful_binary` — 1 if `helpful_vote > 0`, else 0

---

## Repository Structure

| Branch | Owner | Contents |
|---|---|---|
| `feat/multimodel` | Sana | Data pipeline, multimodal model (text + image) |
| `feat/lstm` | Lithika | LSTM sequence model |
| `BERT-(Kevin)` | Kevin | BERT-based helpfulness classifier |
| `feat/validation` | Sanath | Validation notebooks and documentation |
| `feat/viz` | Arjun | EDA visualisation notebook |

---

## This Branch — `feat/viz`

Exploratory data analysis and visualisation for the helpfulness prediction task.

### Files

```
notebook/
  viz.ipynb         — Main EDA notebook (32 cells, fully executed)
report/
  label_definition.md  — Target variable definition and class distribution
requirements.txt    — Python dependencies
```

### Notebook Coverage

1. Setup & data load (2,826,526 reviews from `amazon_reviews_s30.parquet`)
2. Target variable analysis — class balance, vote distribution
3. Feature → target relationships — correlation heatmap
4. Text characteristics — length, word count, depth by class
5. Temporal patterns — time-based helpfulness trends
6. Rating deep dive — skew and helpfulness by rating
7. Verified purchase analysis — trust signal impact
8. Correlation & feature overview — Pearson r rankings
9. NLP analysis — top unigrams and bigrams per class (CountVectorizer)
10. Mutual information — non-linear feature importance proxy
11. Feature interaction effects — length × rating × verified
12. Violin plots — distribution by class
13. Temporal train/val/test split (pre-2020 / 2020 / post-2020)
14. Split export — `splits/train.parquet`, `val.parquet`, `test.parquet`
15. Modelling implications — class imbalance, feature selection notes

### Key Findings

- `word_count` and `review_length` are the strongest predictors (r ≈ +0.24)
- Verified purchases are slightly *less* likely to be helpful (r ≈ −0.08)
- Older reviews accumulate more votes — temporal split preferred over random
- Class imbalance ~53/47 on the filtered dataset → `class_weight='balanced'` recommended
- Top MI features: `word_count`, `review_length`, `is_long`, `year`
