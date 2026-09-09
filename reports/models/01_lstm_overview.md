# LSTM v1 High-Level Overview

## Goal

Predict whether an Amazon review is helpful (`helpful` = 1) using both:
- review text, and
- structured metadata (rating, length, verification, popularity, etc.).

## End-to-End Flow

1. Load processed review data and define the binary target.
2. Clean and normalize text (remove HTML/URLs/symbol noise, keep `[SEP]` marker).
3. Split data into train/test before vocabulary building to avoid leakage.
4. Build a train-only vocabulary and convert text into fixed-length token IDs.
5. Prepare optional GloVe embeddings for richer token initialization.
6. Train traditional TF-IDF baselines (LogReg, Naive Bayes, Random Forest).
7. Train deep sequence models:
   - Vanilla LSTM (text only),
   - BiLSTM + GloVe (text only),
   - Hybrid BiLSTM (text + metadata).
8. Evaluate with Accuracy, F1, Precision, Recall, and AUC.
9. Add explainability with LIME (text-local) and SHAP (metadata influence).

## Model Strategy

- **Baselines first**: establish a reliable performance floor.
- **Sequence modeling next**: capture word order/context not available in TF-IDF.
- **Hybrid fusion last**: combine linguistic and behavioral signals for best lift.

## Inputs Used by the Hybrid Model

- **Text branch**: tokenized and padded review/title text.
- **Metadata branch**: normalized numeric features such as:
  - rating,
  - review length,
  - title-to-text ratio,
  - verification flag,
  - popularity and temporal features.

## Typical Outputs

- Comparison table of all models and key metrics.
- Training curves for deep models.
- Confusion matrix and classification report.
- LIME token-level local explanations.
- SHAP global/local feature attributions for metadata.

## Practical Use

Use the final score as a ranking or prioritization signal (for triage/surfacing), then tune threshold and calibration based on your precision/recall needs.
