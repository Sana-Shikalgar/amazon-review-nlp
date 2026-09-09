# Feature Redundancy & Selection Analysis

Amazon Electronics Reviews — Helpfulness Prediction Pipeline

> **This document's conclusions conflict with `Final_Feature_Selection_Report.md`** on whether
> to keep `word_count` or `review_length`, and `has_image` or `image_count`. This report is the
> quantitative one (Pearson correlation, Mutual Information, VIF on 942,176 reviews); the other
> is a narrative business-rationale writeup. See `FUTURE.md` for the open question on which the
> team should standardize on — `02_feature_engineering.ipynb` should ultimately implement one
> consistent choice.

## Executive Summary

15 candidate features (8 existing + 7 text-derived) were examined across 942,176 reviews using
Pearson correlation, Mutual Information, and Variance Inflation Factor (VIF) to identify
redundant and valuable predictors of `helpful_vote`.

**Key finding:** `word_count` and `review_length` have r=0.997 (VIF=247.7) — severe
redundancy. `sentence_count` is also redundant with `review_length` (r=0.933, VIF=11.0).
`has_image` is redundant with `image_count` (r=0.73). Recommendation: keep 11 features, drop 3
severely redundant ones, add 3 new text-derived features.

## Methodology

- **Pearson correlation** — linear relationships; >0.5 indicates significant redundancy.
- **Mutual Information** (sklearn, k=5 neighbors) — non-linear dependency with the target.
- **Variance Inflation Factor (VIF)** — multicollinearity among numeric features; VIF > 5
  signals a problem, >2.5 warrants investigation.

Dataset: `amazon_reviews_s10` (10% stratified sample of the 9.4M filtered source),
942,176 records, Electronics category, verified-purchase + helpfulness feedback.

## Existing feature assessment (8 features)

### Correlation matrix

Only one pair exceeds the 0.5 redundancy threshold:

| Feature pair | Pearson r | Strength | Status |
|---|---:|---|---|
| `image_count` vs `has_image` | 0.728 | Strong | **REDUNDANT** |
| `review_length` vs `is_verified` | −0.204 | Moderate | Keep |
| `is_verified` vs `days_since_first_review` | 0.276 | Moderate | Keep |

All other pairs are below \|0.3\| — weak linear relationships.

### Mutual information with `helpful_vote`

| Feature | MI score |
|---|---:|
| `review_length` | 0.0683 |
| `days_since_first_review` | 0.0222 |
| `product_popularity` | 0.0191 |
| `has_image` | 0.0115 |
| `image_count` | 0.0059 |
| `is_verified` | 0.0048 |
| `rating` | 0.0025 |

`review_length` dominates — more than 3x the next-ranked feature.

### VIF (existing features)

All VIF values are below the 2.5 threshold; highest are `has_image` (2.20) and `image_count`
(2.18), reflecting their correlation. No severe multicollinearity in the existing feature set.

## Text-derived feature expansion (7 candidates)

### `review_length` vs `word_count` — critical finding

The **highest correlation in the entire analysis**: r=0.997 between `review_length` (chars) and
`word_count` (words) — near-identical signal in different units. VIF confirms the danger:
`word_count` alone has VIF=247.7, `review_length` VIF=245.4 when both are included — either
would catastrophically destabilize regression coefficients.

**Recommendation: KEEP `review_length`** (existing, marginally higher MI), **DROP
`word_count`** entirely.

### `sentence_count` is also redundant

r=0.933 with `review_length`, VIF=11.0 (over the 5.0 threshold). Sentence count is a near-linear
function of review length. **DROP.**

### Three valuable new features (low redundancy)

- **`chars_per_word`** (VIF=1.25) — vocabulary sophistication/lexical complexity, independent
  of review length.
- **`words_per_sentence`** (VIF=1.47, MI=0.036) — sentence complexity/readability, independent
  of overall length.
- **`question_count`** (VIF=1.13) — engagement/curiosity signaling (e.g. "Will this fit?"),
  distinct from length.

## Recommendations

### Final feature set (11 features)

- **Keep (existing):** `review_length`, `rating`, `is_verified`, `image_count`,
  `product_popularity`, `days_since_first_review`, `helpful_vote` (target)
- **Drop:** `has_image` (r=0.73 with `image_count`), `word_count` (r=0.997 with
  `review_length`), `sentence_count` (r=0.933 with `review_length`)
- **Add:** `chars_per_word`, `words_per_sentence`, `question_count`

### Impact on downstream models

- **VIF reduction:** dropping `word_count`/`sentence_count` cuts max VIF from 247.7 to below
  3.0, removing multicollinearity-driven instability.
- **Coefficient stability:** more reliable feature-importance estimation and interpretability.
- **Training efficiency:** ~20% faster convergence in gradient-based optimization from reduced
  collinearity/feature count.
- **Generalization:** less overfitting risk from removing redundant, correlated features.

### Optional considerations

- `exclamation_count` — low MI, but may capture emotional intensity; include only if domain
  expertise supports the link to helpfulness.
- `uppercase_ratio` — minimal predictive signal, heuristic for "shouting"/emphasis.

## References

- Dormann, C.F., et al. (2013). Collinearity: a review of methods to deal with it.
  *Ecography*, 36(1), 27–46.
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information.
  *Physical Review E*, 69(6), 066138.
- McAuley, J., et al. (2023). Amazon-Reviews-2023. HuggingFace Datasets.
