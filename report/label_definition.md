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

## Class Distribution (on `amazon_reviews_s30.parquet`)

> **Note — label adaptation:** `amazon_reviews_s30.parquet` was pre-filtered to retain only
> reviews with `helpful_vote > 0`, so the primary threshold produces a single-class label.
> `05_visualization.ipynb` detects this and falls back to a **median split** on `helpful_vote` (threshold = 1),
> producing the near-balanced distribution below.

| Class         | Label | Actual % (2,826,526 rows) |
|---------------|-------|--------------------------|
| Not Helpful   | 0     | ~53%                     |
| Helpful       | 1     | ~47%                     |

Median split: `helpful_binary = 1` if `helpful_vote > 1`, else `0`.

*Exact figures printed in `05_visualization.ipynb` Cell 3 output.*

## Implication for Modelling

- Use `class_weight='balanced'` or focal loss to handle imbalance.
- Primary metric: **F1-score (macro)** and **AUC-ROC** — not accuracy.
- Stratify any random splits on `helpful_binary` if temporal split is not used.
