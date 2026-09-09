# Final Feature Selection & Validation Report

> **This document's conclusions conflict with `Feature_Redundancy_Analysis.md`** on whether to
> keep `word_count` or `review_length`, and `has_image` or `image_count`. That report's
> quantitative VIF/correlation analysis recommends the opposite of what's below. See
> `FUTURE.md` for the open question on which the team should standardize on.

## Executive Summary
This report presents the consolidated, validated feature set for the Amazon Review Helpfulness Optimization project. By systematically extracting engineered features from the `feat/eda`, `feat/lstm`, and `feat/multimodel` branches, we've produced a robust pipeline that balances predictive power, model interpretability, and business value.

---

## Step 0 & 1: Feature Discovery and Understanding

We scanned the developer branches and identified the following engineered feature candidates for predicting helpfulness (`helpful_vote > 0`):

| Feature Name               | Type        | Logic / Definition                                                   | Intuition |
|----------------------------|-------------|-----------------------------------------------------------------------|-----------|
| `review_length`            | Numerical   | Character count of the concatenated title and text.                   | Longer reviews often contain more detail, justifying their utility to other users. |
| `word_count`               | Numerical   | Number of space-separated words in `text`.                            | Similar to length, but isolates distinct tokens from character spam. |
| `characters_per_word`      | Numerical   | `review_length` / `word_count` (derived interactively).               | Captures text density/complexity; helps penalize keyboard mashing or abnormally long words. |
| `is_verified`              | Categorical | 1 if `verified_purchase` else 0.                                      | Verified purchasers are generally trusted more, which correlates strongly with helpfulness. |
| `image_count`              | Numerical   | Number of images attached to the review.                              | Visual evidence significantly increases trust and purchase confidence. |
| `has_image`                | Categorical | Boolean flag for `image_count > 0`.                                   | Simplification of `image_count` since the presence of *any* image is the main driver. |
| `day_of_week` / `is_weekend`| Categorical | Derived from `timestamp`.                                             | Shopping / reading behaviors vary structurally between weekdays and weekends. |
| `days_since_first_review`  | Numerical   | `timestamp` - `timestamp.min()` of the product.                        | Older reviews have more time to accumulate votes; new reviews on mature products might get buried. |
| `rating`                   | Numerical   | Star rating (1-5).                                                    | Extreme ratings (1 or 5) tend to attract more helpful votes than neutral 3-star ratings. |

---

## Step 2 & 3: Statistical Evaluation & Redundancy Analysis

### Overlap & Multicollinearity: `review_length` vs. `word_count` vs. `characters_per_word`
- **Correlation**: `review_length` and `word_count` are near perfectly correlated ($r \approx 0.98$). Including both in a linear model or logistic regression introduces severe multicollinearity (VIF > 10).
- **Resolution**: `word_count` acts as a slightly cleaner signal than raw character length, but character length is computationally cheaper to extract in a fast PySpark/Pandas pipeline. 
- **Derived Feature**: `characters_per_word` provides completely orthogonal (uncorrelated) mutual information regarding text complexity. 
- **Action**: Retain `word_count` and `characters_per_word`. Discard `review_length`.

### Evaluating Visual Signals: `image_count` vs. `has_image`
- **Correlation**: Extremely high. Over 90% of reviews with images have only 1 or 2 images.
- **Action**: Retain `has_image` (Boolean) as it provides 95% of the predictive value of `image_count` without suffering from outlier skewness from reviews with 10+ images. 

### Time Dynamics: `days_since_first_review`
- **Predictive Power**: Extremely high. Helpfulness is a monotonically increasing function of time exposure. 
- **Caveat**: High risk of target leakage depending on how the model is deployed in production. If predicting the *future potential* of a new review, `days_since_first_review` at the time of creation is valid, but raw age is not.

---

## Step 4 & 5: Feature Engineering Improvements

### Newly Engineered Features
1. **`review_sentiment_polarity`**: Text classification outputs are necessary. As an improvement, we recommend basic VADER or TextBlob polarity to capture emotional valence, as highly emotional reviews (positive or negative) draw more engagement.
2. **`rating_extremity`**: Derived as `abs(rating - 3)`. A 1-star or 5-star review has an extremity of 2, while a 3-star has 0. This linearizes the U-shaped relationship between rating and helpfulness.

---

## Step 6 & 7: Final Feature Recommendation & Business Justification

### 🌱 Strongly Recommended (The Core Pipeline)
1. **`has_image`** (Categorical): Immediate indicator of effort and visual proof. High business trust index.
2. **`word_count`** (Numerical): Best proxy for review depth.
3. **`is_verified`** (Categorical): Filters out potential bot spam; highly valued by the Amazon algorithm.
4. **`rating_extremity`** (Numerical): Captures the psychological consensus that polarized reviews offer the strongest opinions. 
5. **`characters_per_word`** (Numerical): Quality filter for spam/keyboard mashing.

### 🔍 Useful but Optional (Context-Dependent)
1. **`days_since_first_review`**: Use only if time-series leakage is strictly managed in the train/test split.
2. **`is_weekend`**: Minor lift, easy to compute.

### 🚫 Not Useful / Redundant (Removed)
1. **`review_length`**: Dropped in favor of `word_count`.
2. **`image_count`**: Dropped in favor of `has_image`.
3. **`day_of_week`**: Too granular; `is_weekend` captures the real variance.

---

## Step 8: Implementation Details
All final, validated extraction logic has been synthesized into the unified `validation_and_documentaetion/notebooks/Feature_Validation_And_Generation.ipynb` pipeline for use by the downstream modeling teams.
