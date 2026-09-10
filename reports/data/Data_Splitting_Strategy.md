# Data Splitting Strategy: Train/Test Protocol

Amazon Electronics Reviews — Helpfulness Prediction Pipeline

## Executive Summary

The team standardizes on an **80/20 stratified random split** (`test_size=0.20`,
`random_state=42`, stratified on `rating`) as the primary evaluation protocol for all
individual models. It maximizes training data (2.26M records on the full `s30` dataset) while
providing a statistically robust test set (565K records) with confidence intervals narrower
than ±1 percentage point on standard metrics.

A **temporal hold-out on 2023 reviews** is endorsed as a secondary diagnostic tool to detect
distribution shift — not the primary evaluation metric.

All team members must use the same split for experimental reproducibility and model
comparability.

## Splitting strategies evaluated

### 1. Random shuffled split (80/20) — chosen

Randomly assign 80%/20%, stratified on a target-derived variable to preserve marginal
distribution.

- **Pros:** simple, reproducible, maximizes training data, standard for i.i.d. cross-sectional
  data, statistically robust test set.
- **Cons:** assumes i.i.d. observations (may not hold if review patterns evolve over time), risk
  of temporal leakage, doesn't directly test generalization to future data.
- **Academic backing:** Hastie, Tibshirani & Friedman (2009) recommend 80/20 for datasets
  >100K records; Raschka (2018) shows empirically it gives the best bias-variance tradeoff for
  N > 1M.

### 2. Temporal split (train: pre-2023, test: 2023)

Simulates real-world deployment: train on historical data, test on genuinely future data.

- **Pros:** no temporal leakage, realistic production performance estimate, detects
  distribution shift.
- **Cons:** uneven split sizes by review volume per year, 2023 may have its own distribution
  shift (new products, platform changes), not standard for cross-sectional tasks.
- **Academic backing:** Bergmeir & Benitez (2012) argue temporal validation is mandatory for
  time-series-like data; Cerqueira, Torgo & Mozetic (2020) show random splits can overestimate
  generalization on temporal data.

### 3. Group-based split (by product / `parent_asin`)

All reviews for a given product go entirely to train or test.

- **Pros:** tests generalization to unseen products, eliminates product-specific leakage.
- **Cons:** highly uneven group sizes, complex to implement correctly, reduces test-set
  diversity for rare products. Better suited to specialized out-of-distribution studies than as
  the primary split.

## Why 80/20, not 70/30 or 90/10?

On `s30` (2.8M records): 80/20 yields 2,261,221 train / 565,305 test.

- **vs. 70/30** (1,978,368 / 848,158): the extra ~283K test records improve CI width by
  <0.2 percentage points — not worth sacrificing that much training data.
- **vs. 90/10** (only ~282,653 test records): insufficient for reliable per-class metrics on a
  multi-class task; higher variance on rare patterns.
- Guyon (1997) and Hastie et al. (2009) converge on 80/20 as optimal for datasets >100K records.

## Stratification variable

Stratify on **`rating`** (1–5 stars) — Amazon reviews are rating-imbalanced (5-star
over-represented, 2-star under-represented), so stratifying preserves this in both splits.
`helpful_vote` is too skewed (median=1, long tail) for direct stratification; binning into
quartiles is a documented alternative if the team prioritizes helpfulness-target balance instead.

## Implementation

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=df['rating']
)
```

## Temporal validation (secondary diagnostic)

After training on the 80/20 split, also evaluate on all 2023 reviews (regardless of original
partition). If temporal performance degrades by more than 5% on key metrics (R²/MAE), this
signals distribution shift and should be documented in the model report. Degradation under 5%
indicates reasonable robustness.

## Data leakage prevention

- Never use the test set for feature engineering, hyperparameter tuning, or threshold selection
  — use the training set or cross-validation within it only.
- Fit all scaling/normalization/encoding on training data only; apply (don't refit) to test.
- `helpful_vote`-derived features (log-transform, binning) are fine if computed before the
  split and don't directly copy the target.
- Product-level aggregates (e.g. `product_popularity`) computed on the full dataset before
  splitting are acceptable — they're static metadata, not target-derived.
- Never build NLP vocabulary/embeddings using test-set text — fit on training data only.

## Cross-validation protocol

For hyperparameter tuning, use 5-fold stratified CV **within the training set only**:

1. Split the training set (80%) into 5 stratified folds on `rating`.
2. For each hyperparameter candidate, train on 4 folds, evaluate on the held-out fold; repeat
   for all 5 and compute mean ± std CV performance.
3. Select the best mean-CV configuration.
4. Train a final model on the full training set with those hyperparameters.
5. Evaluate once on the test set (20%) and report that as final performance.

Report both CV performance and final test performance in model reports. Never use the test set
during CV.

## References

- Bergmeir, C., & Benitez, J.M. (2012). On the use of cross-validation for time series
  predictor evaluation. *Information Sciences*, 191, 192–213.
- Cerqueira, V., Torgo, L., & Mozetic, I. (2020). Evaluating time series forecasting models.
  *Machine Learning*, 109(11), 1997–2028.
- Guyon, I. (1997). A scaling law for the validation-set training-set size ratio.
  AT&T Bell Laboratories.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
  (2nd ed.). Springer.
- Kaufman, S., et al. (2012). Leakage in data mining: formulation, detection, and avoidance.
  *ACM TKDD*, 6(4).
- Raschka, S. (2018). Model evaluation, model selection, and algorithm selection in machine
  learning. arXiv:1811.12808.
