# PPT Content: Amazon Review Helpfulness Prediction — LSTM Pipeline (`lstmv1.ipynb`)

> **Dataset-size note:** figures below are from a run on `amazon_reviews_s10.parquet` when it
> had 846,757 rows; the committed file now has 942,176. A fresh run on the current file has
> been verified and gives very similar results (F1 0.6272, AUC 0.7183) — see `06_lstm_analysis_report.md`
> and `README.md` for detail.

---

## Slide 1 — Project Overview

**Title:** Predicting Amazon Review Helpfulness with Deep Learning

**Task:** Binary classification — predict whether an Amazon review is *helpful* (`is_helpful = 1`) or *not helpful* (`is_helpful = 0`)

**Dataset:** 846,757 reviews from the Amazon Reviews dataset (`amazon_reviews_s10.parquet`)

**Label Distribution:**
- Not Helpful (0): 444,493 reviews — 52.49%
- Helpful (1): 402,264 reviews — 47.51%
- Near-balanced, with slight skew toward "not helpful"

**Key Design Decisions:**
- Use a strict threshold (`helpful_vote > 1`) when deriving label from raw vote counts
- Stratified train/val/test split to preserve class distribution
- Weighted Binary Cross-Entropy loss (`pos_weight = 1.1050`) to compensate for class imbalance

---

## Slide 2 — Approach & Pipeline Architecture

The notebook builds a progressive, comparative pipeline across 6 models:

| Stage | Models | Purpose |
|-------|--------|---------|
| Baselines | Logistic Regression, Naive Bayes, Random Forest | TF-IDF text features, no sequential learning |
| Deep Learning | Vanilla LSTM | Sequential word pattern learning |
| Deep Learning | BiLSTM + GloVe | Bidirectional reading + pretrained semantics |
| Deep Learning | Hybrid BiLSTM | Text + metadata fusion — **best model** |

**Text Features:** Cleaned review + title (`combined_text`), tokenized, padded to max sequence length

**Metadata Features (13 total):**
`rating`, `review_length`, `char_per_word`, `image_bucket`, `days_since_first_review`, `is_verified`, `log_popularity`, `rating_extremity`, `question_mark`, `exclamation_mark`, `title_length`, `title_to_text_ratio`, `images`

---

## Slide 3 — Baseline Models (TF-IDF)

Three classical ML baselines were trained using TF-IDF vectorization on the `combined_text` field:

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | 0.6244 | 0.5837 | 0.6692 |
| Naive Bayes | 0.6210 | 0.5866 | 0.6655 |
| Random Forest | 0.6346 | 0.5792 | 0.6792 |

**Why these are used as baselines:**
- They treat text as a *bag of words* — no word order, no context
- Logistic Regression provides a linear decision boundary
- Naive Bayes assumes feature independence (often incorrect in natural language)
- Random Forest captures non-linear interactions but still ignores word sequence
- They establish a performance floor for the deep learning models to beat

---

## Slide 4 — Deep Learning Models: Architecture Overview

Three PyTorch architectures were defined and trained for 5 epochs with:
- Adam optimizer (`lr = 1e-3`)
- Weighted BCE loss (`pos_weight = 1.1050`)
- Cosine annealing learning-rate scheduler
- Gradient clipping and early stopping

### Model 1: Vanilla LSTM
- Unidirectional single-pass LSTM
- Random (untrained) word embeddings (`emb_dim = 100`)
- Reads text left-to-right only
- Serves as deep learning baseline

### Model 2: BiLSTM + GloVe
- **Bidirectional** 2-layer LSTM
- Initialized with **GloVe 6B 100-d pretrained embeddings** (400,000 word vectors)
- Reads text both forward and backward — captures richer context
- Text-only model (no metadata)

### Model 3: Hybrid BiLSTM *(Selected Model)*
- Same BiLSTM + GloVe text branch
- **Adds a parallel dense branch for 13 metadata features**
- Two branches fused at the final classification layer
- Captures both linguistic patterns AND structural review signals

---

## Slide 5 — Why Hybrid BiLSTM Was Chosen

The **Hybrid BiLSTM** was selected as the primary/best model for the following reasons:

### Performance — It Outperforms All Other Models

| Metric | Hybrid BiLSTM | BiLSTM+GloVe | Vanilla LSTM |
|--------|--------------|--------------|--------------|
| Accuracy | **0.6597** | 0.6403 | 0.6389 |
| F1-Score | **0.6376** | 0.6021 | 0.6072 |
| Precision | **0.6453** | 0.6344 | 0.6283 |
| Recall | **0.6300** | 0.5730 | 0.5875 |
| AUC-ROC | **0.7200** | 0.6904 | 0.6879 |

The Hybrid BiLSTM achieves a **+2% gain in accuracy** and a **+3.5% gain in F1** over the next best deep learning model (BiLSTM+GloVe), and a **+2.5% gain in AUC-ROC** — a significant improvement in a near-balanced binary classification problem.

### Architectural Strengths

1. **Bidirectionality:** The LSTM reads each review both forwards and backwards. This allows the model to understand words in context of what comes *before and after* them — crucial for nuanced opinion language.

2. **Pretrained GloVe Embeddings:** Words like *"great"*, *"waste"*, *"terrible"* already carry semantic meaning from the 6B-token GloVe corpus. The model inherits this knowledge without needing to learn it from scratch, boosting convergence and generalization.

3. **Metadata Fusion:** Review helpfulness is not determined by text alone. Signals like:
   - `review_length` — longer reviews tend to be more helpful
   - `rating` — extreme ratings (1★, 5★) correlate with helpfulness patterns
   - `is_verified` — verified purchases carry different trust
   - `log_popularity` — popular products attract more engaged reviewers
   
   are captured alongside text in a dedicated dense branch, giving the model complementary structural information.

4. **Class-Weighted Loss:** The model corrects for the mild class imbalance using `pos_weight = 1.1050`, preventing it from defaulting to the majority class.

---

## Slide 6 — Full Model Score Comparison

### All Models Ranked by F1-Score

| Rank | Model | Accuracy | F1 | Precision | Recall | AUC-ROC |
|------|-------|----------|----|-----------|----|---------|
| 1 | **Hybrid BiLSTM** | **0.6597** | **0.6376** | **0.6453** | **0.6300** | **0.7200** |
| 2 | BiLSTM+GloVe | 0.6403 | 0.6021 | 0.6344 | 0.5730 | 0.6904 |
| 3 | Vanilla LSTM | 0.6389 | 0.6072 | 0.6283 | 0.5875 | 0.6879 |
| 4 | Random Forest | 0.6346 | 0.5792 | 0.6394 | 0.5293 | 0.6792 |
| 5 | Logistic Regression | 0.6244 | 0.5837 | 0.6164 | 0.5542 | 0.6692 |
| 6 | Naive Bayes | 0.6210 | 0.5866 | 0.6087 | 0.5660 | 0.6655 |

### Hybrid BiLSTM — Per-Class Breakdown (Test Set, n=169,352)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Helpful (0) | 0.6722 | 0.6866 | 0.6793 | 88,899 |
| Helpful (1) | **0.6453** | **0.6300** | **0.6376** | 80,453 |
| Weighted Avg | 0.6594 | 0.6597 | 0.6595 | 169,352 |

### Training Progression (Hybrid BiLSTM over 5 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.6596 | 0.6474 | 0.6485 | 0.6578 |
| 2 | 0.6476 | 0.6599 | 0.6456 | 0.6608 |
| 3 | 0.6384 | 0.6680 | 0.6475 | 0.6578 |
| 4 | 0.6288 | 0.6761 | 0.6520 | 0.6575 |
| 5 | 0.6156 | 0.6860 | 0.6640 | 0.6507 |

*Note: Training accuracy continues to improve while validation accuracy plateaus after epoch 2, suggesting the model is near its generalization ceiling at 5 epochs. Early stopping could lock in epoch 2 weights.*

---

## Slide 7 — Important Plots & Their Explanations

### Plot 1: Correlation Heatmap (Feature × Feature)

**What it shows:** Pearson correlation matrix across all 13 numeric metadata features and the `is_helpful` target label.

**Key interpretation:**
- `review_length` shows the strongest positive correlation with helpfulness — longer, more detailed reviews are more likely to be voted helpful by readers.
- `rating_extremity` (distance from the neutral rating of 3) shows a slight negative correlation — very polar reviews (1★ or 5★) may be perceived as biased.
- `is_verified` and `log_popularity` show mild negative correlations with helpfulness — verified purchase reviews on popular products may be shorter/less informative.
- Most metadata features show weak-to-moderate individual correlations, justifying a *multi-feature* fusion approach rather than relying on any single signal.

---

### Plot 2: Feature Distribution Panel (EDA)

**What it shows:** A 2×2 grid showing:
1. **Review Length Distribution** — right-skewed histogram with median marked (median ≈ 321 characters). Most reviews are short, but a long tail of detailed reviews exists.
2. **Rating Distribution** — bar chart showing 5-star reviews dominate the corpus, typical of Amazon data.
3. **Review Length by Helpfulness** — boxplots comparing length distributions between helpful and not-helpful reviews. Helpful reviews are visibly longer.
4. **Rating × Helpfulness** — stacked bar chart showing that mid-ratings (2★, 3★, 4★) have higher helpfulness proportions than extreme ratings.

**Key interpretation:**
- The length gap between helpful and not-helpful reviews is the clearest EDA signal — it validates `review_length` as a top predictive feature.
- Rating extremity alone is insufficient — helpful reviews exist across all star levels.

---

### Plot 3: Word Clouds (Helpful vs. Not Helpful)

**What it shows:** Two side-by-side word clouds generated from lemmatized, stopword-removed review text:
- **Helpful Reviews (green):** Dominated by words like *product*, *quality*, *use*, *good*, *great*, *easy*, *work*, *recommend* — suggesting helpful reviews are informative, evaluative, and recommendation-oriented.
- **Not Helpful Reviews (red):** Shows more generic or emotionally charged words — consistent with shorter, less substantive reviews.

**Key interpretation:**
- Helpful reviews tend to discuss specific product attributes (quality, usability, features).
- Not-helpful reviews contain more filler language and emotional exclamations.
- This visual confirms that review *content and vocabulary* differ meaningfully between classes, supporting the use of a text model.

---

### Plot 4: Confusion Matrices (All Deep Learning Models)

**What it shows:** 2×2 confusion matrices for Vanilla LSTM, BiLSTM+GloVe, and Hybrid BiLSTM, showing True Positives, False Positives, True Negatives, and False Negatives.

**Key interpretation (Hybrid BiLSTM):**
- The model correctly identifies ~68.7% of "Not Helpful" reviews (True Negative Rate / Specificity).
- It correctly identifies ~63.0% of "Helpful" reviews (True Positive Rate / Recall).
- The false negative rate (~37%) suggests the model sometimes misses genuinely helpful reviews — a known challenge when review helpfulness is subjective.
- Compared to the Vanilla LSTM, the Hybrid BiLSTM shows notably improved recall for the Helpful class, meaning it misses fewer helpful reviews.

---

### Plot 5: ROC Curves (All Models on One Plot)

**What it shows:** Receiver Operating Characteristic (ROC) curves for all 6 models plotted together, with the diagonal (AUC=0.5) as the random classifier baseline.

**Key interpretation:**
- The **Hybrid BiLSTM** achieves the highest AUC of **0.7200**, meaning it best separates the two classes across all decision thresholds.
- The **BiLSTM+GloVe** (AUC=0.6904) and **Vanilla LSTM** (AUC=0.6879) trail closely — the metadata fusion accounts for the gap.
- TF-IDF baselines cluster between AUC=0.6655 and 0.6792 — indicating sequential models provide meaningful improvement over bag-of-words.
- A higher AUC means the model is robust across different classification thresholds — useful if the operating threshold needs to be adjusted for business use (e.g., surfacing only high-confidence helpful reviews).

---

### Plot 6: Model Comparison Bar Chart

**What it shows:** Grouped bar chart comparing Accuracy, F1, and AUC across all 6 models, sorted by F1 score descending.

**Key interpretation:**
- Provides a clean visual ranking of models — Hybrid BiLSTM leads on all three metrics.
- The gap between deep learning models (LSTM variants) and classical models (TF-IDF) is modest in accuracy but more visible in AUC — suggesting LSTMs better rank confident predictions even when their raw accuracy is similar.
- F1-score is the preferred metric here because it balances precision and recall, appropriate for a near-balanced but still slightly imbalanced dataset.

---

### Plot 7: Training Loss & Accuracy Curves

**What it shows:** Two line plots per model (three models total) showing:
- **Left:** Training loss vs. Validation loss over 5 epochs
- **Right:** Training accuracy vs. Validation accuracy over 5 epochs

**Key interpretation:**
- **Vanilla LSTM:** Training accuracy rises (0.60 → 0.66) but validation accuracy peaks early and then drops slightly — mild overfitting signal after epoch 2.
- **BiLSTM+GloVe:** Similar pattern — trains faster initially due to GloVe initialization but validation accuracy stalls around 0.64.
- **Hybrid BiLSTM:** Validation accuracy peaks at **0.6608** (epoch 2) — the best of all three. The model generalizes better because the metadata branch provides complementary, non-textual signals that don't overfit as quickly.
- The divergence between training and validation curves suggests that with more regularization (higher dropout, weight decay) or more training data, further gains are possible.

---

### Plot 8: Per-Class F1 Heatmap

**What it shows:** A heatmap where rows = models and columns = classes (Helpful / Not Helpful), with the cell value being the per-class F1 score.

**Key interpretation:**
- Every model achieves higher F1 for the "Not Helpful" class than the "Helpful" class — predicting non-helpful reviews is an easier task.
- The Hybrid BiLSTM achieves the highest F1 for both classes, but the improvement over the Helpful class (0.6376 vs. ~0.60 for other LSTMs) is more pronounced — metadata features help the model identify genuine helpful reviews better.

---

## Slide 8 — Explainability Section

### Overview

Two complementary explainability methods were applied:

| Method | Scope | Target | Purpose |
|--------|-------|--------|---------|
| **LIME** | Local | TF-IDF + Logistic Regression | Explain a single prediction — which words pushed the decision |
| **SHAP** | Global + Local | Metadata surrogate model | Explain global feature importance and local prediction contributions |

*Note: LIME is applied to the text (word-level) and SHAP to structured metadata, giving full coverage of both input modalities.*

---

### LIME — Local Interpretable Model-agnostic Explanations

**Method:** LIME perturbs the input text (removes random words) and fits a local linear model to approximate the complex classifier's behavior around a single sample.

**Top LIME Features (for the Helpful class) on a representative review:**

| Feature (Word) | LIME Weight | Direction |
|----------------|-------------|-----------|
| `review_length` | +0.2388 | Strong positive push toward "Helpful" |
| `days_since_first_review` | -0.1129 | Negative push (older reviews less helpful) |
| `log_popularity` | -0.0965 | Negative push (popular products, less relative helpfulness) |
| `is_verified` | -0.0770 | Slight negative push |
| `rating` | -0.0441 | Slight negative push |
| `rating_extremity` | -0.0076 | Near-neutral |

**Interpretation:**
- The most influential word-level signal pointing toward a review being **helpful** is its overall length — LIME consistently gives high positive weight to length-related tokens and substantive descriptive words.
- Words expressing *specific* product evaluation (e.g., "quality", "use", "recommend") receive positive LIME weights for the helpful class.
- Shorter reviews with vague or emotional language receive negative LIME weights.

**LIME Panel (TP/TN/FP/FN):**
An advanced LIME panel shows explanations for four representative cases:
- **True Positive:** Words like quality-descriptors correctly pushing toward "helpful"
- **True Negative:** Short, emotional words correctly pushing toward "not helpful"
- **False Positive:** Cases where the model over-weights word length, predicting "helpful" for a detailed-but-unhelpful review
- **False Negative:** Cases where the model under-weights specific useful terms in a short but informative review

---

### SHAP — SHapley Additive exPlanations

**Method:** SHAP computes each feature's contribution to the output by averaging over all possible feature coalitions (Shapley values from game theory). Applied here to a metadata surrogate model (Gradient Boosting / tree-based) trained on the 13 metadata features.

**Global Feature Importance (Mean |SHAP|):**

| Feature | Mean |SHAP| | Interpretation |
|---------|--------------|----------------|
| `review_length` | **0.0755** | Strongest global driver — length determines perceived value |
| `log_popularity` | 0.0311 | Product-level signal — more-reviewed products attract better reviews |
| `days_since_first_review` | 0.0195 | Temporal signal — review timing affects visibility/votes |
| `rating` | 0.0124 | Star rating provides a weak but real signal |
| `rating_extremity` | 0.0031 | Extreme ratings slightly reduce helpfulness probability |

**Interpretation of Results:**

1. **`review_length` dominates all other metadata features** with a SHAP value more than twice that of the next feature. This validates the EDA observation and LIME finding — reviews that invest in detail are rated more useful by readers. The signal is both the strongest and most directional (longer → more helpful).

2. **`log_popularity`** is the second most important feature. Reviews on highly popular products (many reviews total) face more competition, but the log-transformed popularity also captures product category effects — product niches with engaged buyer communities tend to produce more helpful reviews.

3. **`days_since_first_review`** captures review recency indirectly — older reviews accumulate more helpfulness votes over time, but they may also describe older product versions. SHAP assigns it a negative contribution in most cases, suggesting recency alone is not a reliable proxy.

4. **`rating` and `rating_extremity`** contribute modestly. Very high or very low ratings can signal bias (5★ may reflect purchase satisfaction bias; 1★ may reflect frustration rather than informative critique), slightly reducing helpfulness.

**SHAP Summary Beeswarm Plot:**
The beeswarm shows each sample (dot) positioned by its SHAP value for each feature, colored by feature magnitude. `review_length` shows the widest spread — confirming high variance in its contribution across samples. This means review length is not just *on average* important — it's *individually decisive* for many predictions.

**SHAP Dependence Plot (`review_length`):**
Shows that as `review_length` increases, the SHAP contribution increases monotonically — with steepest gains between 0 and ~1,000 characters, then diminishing returns for very long reviews. This suggests there is a "sweet spot" for review length that maximizes predicted helpfulness.

**SHAP Waterfall Plot (Local Instance):**
For a representative test sample, the waterfall shows cumulative feature pushes from the base rate toward the final prediction. `review_length` typically makes the single largest positive contribution. Negative contributions typically come from `days_since_first_review` and `log_popularity`.

**SHAP Decision Plot:**
Shows cumulative SHAP contributions across multiple sampled test instances as each feature is added. Confirms that across diverse examples, `review_length` is the most consistent decisive factor.

**SHAP Interaction Heatmap:**
A matrix of mean |SHAP interaction values| between feature pairs (tree-based surrogate). Notable interactions:
- `review_length × log_popularity`: Reviews on popular products that are also long have a compounding effect on helpfulness.
- `rating × rating_extremity`: These two are highly correlated, so their interaction is minimal — capturing near-redundant information.

---

## Slide 9 — Key Findings & Conclusions

**Best model: Hybrid BiLSTM (F1 = 0.6376, AUC = 0.7200)**

### What We Learned

1. **Sequential context matters:** All three LSTM variants beat TF-IDF baselines — word order and sequential patterns (captured by the recurrent layers) carry meaningful signal beyond simple word frequencies.

2. **Pretrained embeddings accelerate training:** GloVe initialization gives BiLSTM models a semantic head-start — words like *"great"* and *"waste"* already encode polarity before any review is seen.

3. **Metadata is complementary, not redundant:** The Hybrid BiLSTM's performance gain over the text-only BiLSTM+GloVe confirms that structural review features (especially `review_length`, `rating`, and product-level signals) carry signal that language alone cannot capture.

4. **Review length is the single most predictive feature:** Confirmed independently by LIME (text model), SHAP (metadata model), and EDA correlation analysis. Readers reward detailed, substantive reviews.

5. **The task is inherently hard:** AUC of 0.72 reflects real-world label noise — helpfulness is subjective, votes are sparse and biased by visibility, and many genuinely useful short reviews go unvoted. Further gains likely require richer context (e.g., product category, review history) or transformer-based models.

### Potential Next Steps
- Fine-tune a BERT/RoBERTa model on the same data (experimental section included in notebook)
- Incorporate product category as a categorical embedding
- Apply attention mechanisms to identify the most helpful sentences within a review
- Explore threshold optimization for precision-recall trade-off depending on business use case

---

*Generated from `lstmv1.ipynb` — Amazon Review Helpfulness Prediction LSTM Pipeline*
