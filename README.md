# WM9B7-AIDL-AmazonReview

**Task:** Binary classification — `helpful_binary` (1 if `helpful_vote > 0`, else 0)  
**Dataset:** 50,000 Amazon Electronics reviews · `amazon_reviews_us_Electronics_v1_00` via HuggingFace

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

---

# Amazon Electronics Reviews — Sentiment & Emotion Analysis

## Project Overview

Group project for the AIDL module (WMG, University of Warwick).
Dataset: McAuley-Lab/Amazon-Reviews-2023 (Electronics category).

This repository contains the work for **Sentiment & Emotion Analysis** and **Explainability**.

---

## Sentiment & Emotion Analysis

### What was done

Built an inference-only pipeline in `notebook/sentiment_analysis.ipynb` using two pre-trained HuggingFace models applied to the Amazon Electronics Reviews dataset.

**Models used:**
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — 3-class sentiment (positive / neutral / negative)
- `j-hartmann/emotion-english-distilroberta-base` — 7-class emotion (joy, anger, sadness, fear, disgust, surprise, neutral)

No model training was performed. Both models are used as-is from HuggingFace Hub.

**Pipeline steps:**

1. Load `data/processed/amazon_reviews_s10.parquet` (846K reviews, 17 columns)
2. Filter to the top-5 most helpful reviews per product, ranked by `helpful_vote` with `review_length` as tiebreaker — yields ~504K reviews
3. Run batched inference on `combined_text` (title + `[SEP]` + review body) with 512-token truncation
4. Store full class probability distributions alongside top-1 labels
5. Save results to `data/processed/sentiment_emotion_results.parquet`

**Evaluation (Step 3):**
- Sentiment vs. star rating alignment (stacked bar chart per rating)
- Confidence distribution of top-1 sentiment prediction (mean: 0.79 vs. 0.33 random-chance baseline for 3-class)
- Emotion label distribution across the inference set
- Manual spot-checks: 3 reviews sampled per emotion label with qualitative interpretation

### Key findings

- Sentiment aligns well with star ratings at the extremes (1-star → negative, 5-star → positive)
- Mean model confidence of 0.79 indicates decisive, non-trivial predictions on the majority of inputs
- `neutral` is the dominant emotion label (~40%), consistent with the informational tone of many product reviews
- Emotion labels are reliable for clear cases (joy in 5-star praise, anger in defect/support complaints) but act as tone signals rather than direct proxies for star rating

---

## Explainability

### What was done

Added token-level explanations for both the sentiment and emotion models using LIME and SHAP,
implemented in the same notebook (`notebook/sentiment_analysis.ipynb`, Step 4).

**Methods:**
- **LIME** (`lime.lime_text.LimeTextExplainer`) — perturbs input text 200 times and fits a local
  linear model to identify which words most influenced the sentiment prediction
- **SHAP** (`shap.Explainer` with `shap.maskers.Text`) — assigns Shapley values to each token,
  showing its marginal contribution to the predicted class score

Both methods are applied to one representative review per sentiment label (positive / neutral / negative).
SHAP is also applied to the emotion model for one sample, explaining the predicted emotion label.

**Key findings:**
- LIME and SHAP agree on the most influential tokens (e.g. *helpful*, *SURPRISINGLY*, *overall* for positive)
- SHAP additionally reveals that negation (*doesn't allow*) and HTML noise (`<br />`) have measurable
  countervailing effects that LIME's word-removal approach misses
- For the emotion model, *surprises* reduces the neutral score despite appearing in "No surprises" —
  a known limitation of token-level attribution where negation context is not captured

---

### Running the notebook

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure data is available
#    (parquet files are not tracked in git — obtain from shared drive)

# 3. Open notebook
jupyter notebook notebook/sentiment_analysis.ipynb
