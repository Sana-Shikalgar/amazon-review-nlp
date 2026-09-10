# Label Definition — Helpfulness Classification

## Target Variable

```
helpful_binary = 1  if  helpful_vote > 0
helpful_binary = 0  if  helpful_vote == 0
```

## Threshold Justification

**Threshold chosen: `helpful_vote > 0` (at least one helpful vote)**

### Why not a higher threshold (e.g. ≥ 5)?
- ~74% of reviews have zero helpful votes — a higher threshold would create extreme class imbalance (>90% negative class), making the problem harder to learn from.
- A single helpful vote from another customer is a meaningful signal — it means at least one person actively read the review and found it valuable enough to click.
- Higher thresholds (≥5, ≥10) conflate helpfulness with *popularity*: older reviews naturally accumulate more votes regardless of quality, introducing temporal bias.

### Why not a continuous regression target?
- The distribution of `helpful_vote` is heavily zero-inflated and right-skewed (median=0, mean driven by outliers).
- A binary label is more stable, interpretable, and directly aligned with the downstream use case: ranking reviews for display.

### Alignment with literature
- McAuley et al. (Amazon review datasets) and subsequent work commonly binarise at `helpful_vote > 0` or use the ratio `helpful_vote / total_vote > 0.5`. We use the simpler threshold as `total_vote` is not available in this dataset.

## Class Distribution (on `s03_filter.parquet`)

> **Note — label adaptation:** the pipeline pre-filters to retain only reviews with
> `helpful_vote > 0`, so the primary threshold produces a single-class label.
> `05_feature_analysis_and_split.ipynb` detects this and falls back to a **median split** on `helpful_vote`
> (threshold = 1), producing a near-balanced distribution.

> **Stale — needs re-verification.** These figures were measured on the retired
> `amazon_reviews_s30.parquet` (2,826,526 rows). The pipeline now produces
> `s03_filter.parquet` (99,627 rows) instead — the class balance should be similar (same
> `helpful_vote > 1` median-split logic) but hasn't been re-measured on the new file.

| Class         | Label | Actual % (2,826,526 rows, pre-fix) |
|---------------|-------|-------------------------------------|
| Not Helpful   | 0     | ~53%                                |
| Helpful       | 1     | ~47%                                |

Median split: `helpful_binary = 1` if `helpful_vote > 1`, else `0`.

*Exact figures printed in `05_feature_analysis_and_split.ipynb` Cell 3 output — re-run to refresh.*

## Implication for Modelling

- Use `class_weight='balanced'` or focal loss to handle imbalance.
- Primary metric: **F1-score (macro)** and **AUC-ROC** — not accuracy.
- Stratify any random splits on `helpful_binary` if temporal split is not used.
