# LSTM v1 Analysis README

This document provides a high-level and critical interpretation of `lstm/lstmv1.ipynb`, including model outcomes, explainability findings, limitations, and recommended next steps.

> **Dataset-size note:** this analysis was run against `amazon_reviews_s10.parquet` when it had
> 846,757 rows. The currently committed file has 942,176 rows (a later resample) — results
> below will shift somewhat on a fresh run. Directional findings (Hybrid BiLSTM wins,
> `review_length` dominates) are expected to hold; treat exact metric values as historical
> until re-verified.

## 1) Objective and Scope

- Build a binary classifier for review helpfulness (`helpful`).
- Use a hybrid architecture combining:
  - text sequence modeling (LSTM variants), and
  - engineered metadata features.
- Compare deep models against TF-IDF baselines.
- Add explainability using LIME (text-local) and SHAP (metadata global/local).

## 2) Dataset and Target Definition

- Source used in notebook: `data/processed/amazon_reviews_s10.parquet`.
- Effective sample size used in run:
  - Total: `846,757`
  - Train: `677,405`
  - Test: `169,352`
- Label source resolved from dataset: `is_helpful` (already binary).
- Observed class balance:
  - Class `0`: `444,493` (`52.49%`)
  - Class `1`: `402,264` (`47.51%`)

Interpretation:
- The label balance is healthy enough that metrics like F1 and AUC are informative without aggressive rebalancing.

## 3) Feature and Modeling Setup

- Text pipeline:
  - cleaned text with tokenizer-built vocabulary
  - vocabulary size: `50,000` (incl. PAD/OOV)
  - train-token coverage by top vocab: `99.60%`
  - max sequence length: `200`
- Metadata features (13):
  - `rating`, `review_length`, `char_per_word`, `image_bucket`,
  - `days_since_first_review`, `is_verified`, `log_popularity`,
  - `rating_extremity`, `question_mark`, `exclamation_mark`,
  - `title_length`, `title_to_text_ratio`, `images`
- Models compared:
  - Baselines: Logistic Regression, Naive Bayes, Random Forest (TF-IDF)
  - Deep: Vanilla LSTM, BiLSTM+GloVe, Hybrid BiLSTM (text + metadata)

## 3.1) Why These Models Were Chosen (Specialty-Driven Design)

The model stack is intentionally progressive: from simple/transparent baselines to higher-capacity deep sequence models. This is important for scientific validity because it shows not only "what wins," but **why additional complexity is justified**.

### A) TF-IDF + Logistic Regression

- Why included:
  - strong, widely accepted linear baseline for text classification,
  - fast to train and easy to calibrate,
  - coefficients are directly interpretable.
- Specialty:
  - captures global lexical directionality (which words push toward helpful/not-helpful).
- Strengths:
  - robust with large sparse features,
  - good bias-variance tradeoff in many NLP tasks.
- Typical weaknesses:
  - limited handling of word order and compositional semantics,
  - cannot model long-range sequence effects.

### B) TF-IDF + Multinomial Naive Bayes

- Why included:
  - very strong low-compute probabilistic baseline,
  - often competitive when token statistics dominate signal.
- Specialty:
  - frequency-based probabilistic scoring with strong smoothing behavior.
- Strengths:
  - fast, stable, and hard to overfit severely,
  - good sanity-check for token-level separability.
- Typical weaknesses:
  - conditional independence assumption is unrealistic for language,
  - weaker when phrase context/order matters.

### C) TF-IDF + Random Forest

- Why included:
  - non-linear baseline to test whether tree ensembles can extract interactions in sparse features.
- Specialty:
  - captures some non-linear decision boundaries without sequence modeling.
- Strengths:
  - useful contrast against linear models.
- Typical weaknesses:
  - high-dimensional sparse TF-IDF is not the natural regime for RF,
  - can underperform gradient/linear methods on pure bag-of-words setups.

### D) Vanilla LSTM (Text Sequence Model)

- Why included:
  - first deep sequential benchmark beyond bag-of-words.
- Specialty:
  - models token order and contextual accumulation over sequence positions.
- Strengths:
  - can learn local-to-mid-range dependencies and phrase dynamics.
- Typical weaknesses:
  - unidirectional reading may miss reverse-context cues,
  - more sensitive to training dynamics and regularization.

### E) BiLSTM + GloVe

- Why included:
  - tests whether bidirectionality + pretrained lexical priors improve generalization.
- Specialty:
  - reads text left-to-right and right-to-left,
  - starts with semantic priors from external embeddings.
- Strengths:
  - richer context encoding than unidirectional LSTM,
  - potentially faster convergence when domain matches pretrained vectors.
- Typical weaknesses:
  - embedding-domain mismatch can reduce gains,
  - added capacity can overfit with limited epochs/tuning.

### F) Hybrid BiLSTM (Text + Metadata Fusion)

- Why included:
  - helpfulness is not purely linguistic; it is also behavioral/structural.
- Specialty:
  - fuses semantic sequence representation with structured review/product signals.
- Strengths:
  - captures complementary modalities (content + context),
  - better suited for operational helpfulness signals than text-only models.
- Typical weaknesses:
  - requires careful feature governance to prevent leakage/bias,
  - slightly higher implementation complexity.

### G) Why this benchmark order matters

- `LR/NB/RF` establish transparent and computationally cheap reference points.
- `Vanilla LSTM` quantifies value of sequence modeling alone.
- `BiLSTM+GloVe` quantifies value of bidirectionality/pretraining.
- `Hybrid BiLSTM` quantifies value of multimodal fusion.

This staging enables **ablation-like interpretation**: each architectural step answers a specific research question, not just a leaderboard comparison.

## 3.2) Model Selection Decision Matrix

Scores are relative within this project context (`Low`, `Medium`, `High`).

| Model | Interpretability | Training Speed | Inference Cost | Data Requirement | Robustness (small tuning changes) | Sequence Understanding | Metadata Fusion Capability | Best Use Case |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | High | High | High | Low-Medium | High | Low | Low | Transparent baseline, quick deployment, score calibration |
| TF-IDF + Naive Bayes | Medium | Very High | Very High | Low | High | Low | Low | Fast probabilistic baseline, token-separability sanity check |
| TF-IDF + Random Forest | Medium | Medium-Low | Medium | Medium | Medium | Low | Medium (tabular easier than sparse text) | Non-linear baseline contrast, interaction probing |
| Vanilla LSTM | Medium-Low | Medium | Medium | Medium-High | Medium | Medium | Low | First deep sequence benchmark beyond bag-of-words |
| BiLSTM + GloVe | Medium-Low | Medium-Low | Medium | Medium-High | Medium-Low | High | Low | Richer contextual text modeling with pretrained priors |
| Hybrid BiLSTM (Text + Metadata) | Low-Medium | Low | Medium-Low | High | Medium | High | High | Best predictive model when helpfulness depends on content + context |

Decision interpretation:
- If your priority is **explainability + speed**, start with Logistic Regression.
- If your priority is **best predictive performance**, Hybrid BiLSTM is the strongest choice in this run.
- If your priority is **resource-constrained inference**, NB/LR remain attractive despite lower ceiling.
- If your priority is **text semantics only**, BiLSTM-family models are more appropriate than TF-IDF baselines.

## 3.3) Training Configuration & Architecture Details

- TF-IDF baselines: max 30,000 features, unigrams + bigrams, sublinear TF scaling.
- GloVe embeddings: `glove.6B.100d` (100-dimensional), aligned to the training vocabulary;
  known words get their GloVe vector, unknown words get small random init, PAD token is
  zero-vector.
- Model parameter counts:
  - Vanilla LSTM: 6,532,225 (random 128-d embeddings, forward-only, no metadata)
  - BiLSTM+GloVe: 5,631,041 (GloVe 100-d, bidirectional, 2 layers, no metadata)
  - Hybrid BiLSTM: 5,649,105 (GloVe 100-d, bidirectional, 2 layers, + metadata branch)
- Training loop: class-weighted `BCEWithLogitsLoss` (pos_weight computed from the label
  imbalance), gradient clipping at `max_norm=1.0`, `ReduceLROnPlateau` (factor 0.5, patience 1),
  early stopping (patience 3) on validation loss, Adam optimizer at `lr=1e-3`, batch size 256
  train / 512 eval.

Note: an earlier analysis pass over this notebook (942,176-row dataset snapshot) reported the
Hybrid BiLSTM as not finishing training, with Vanilla LSTM/BiLSTM+GloVe both scoring
F1 ≈ 0.59–0.60. That run is superseded by the results below (846,757 rows, all three deep
models completed) — the config details above still apply to the current run.

## 4) Main Quantitative Results

From the notebook's comparison table:

| Model | Accuracy | F1 | Precision | Recall | AUC |
|---|---:|---:|---:|---:|---:|
| **Hybrid BiLSTM** | **0.6597** | **0.6376** | **0.6453** | **0.6300** | **0.7200** |
| Vanilla LSTM | 0.6389 | 0.6072 | 0.6283 | 0.5875 | 0.6879 |
| BiLSTM+GloVe | 0.6403 | 0.6021 | 0.6344 | 0.5730 | 0.6904 |
| Naive Bayes | 0.6210 | 0.5866 | 0.6087 | 0.5660 | 0.6655 |
| Logistic Regression | 0.6244 | 0.5837 | 0.6164 | 0.5542 | 0.6692 |
| Random Forest | 0.6346 | 0.5792 | 0.6394 | 0.5293 | 0.6792 |

Interpretation:
- **Hybrid BiLSTM is the strongest model** across all core metrics.
- Gain pattern suggests metadata contributes real signal beyond pure sequence modeling.
- BiLSTM+GloVe did not beat Vanilla LSTM on F1 in this run, indicating either:
  - embedding/domain mismatch,
  - insufficient fine-tuning epochs,
  - or overfitting/optimization instability.

## 5) Critical Analysis of Training Dynamics

- BiLSTM+GloVe training log showed:
  - train accuracy rising (`~0.63 -> ~0.67`)
  - validation accuracy peaking early (`~0.64`) then flattening/declining
  - early stopping at epoch 5

Interpretation:
- This is a mild overfitting signature.
- The model likely learns easy lexical patterns quickly but generalization saturates.
- Hybrid model's improvement implies metadata regularizes decision boundaries and helps recover from purely text-based ambiguity.

## 6) Explainability Findings

### 6.1 LIME (local text explanations)

- Example high-confidence helpful prediction had positive token contributions (e.g., domain/context terms like `app`, `video`, `like`, `offer`).

Interpretation:
- Baseline lexical model is mostly using topical and sentiment-associated cues.
- LIME confirms local plausibility, but local fidelity does not guarantee global causality.

### 6.2 SHAP (metadata surrogate model)

Top reported global drivers (mean absolute SHAP):

1. `review_length` (`0.075473`)
2. `title_to_text_ratio` (`0.036206`)
3. `log_popularity` (`0.031091`)
4. `days_since_first_review` (`0.019462`)
5. `title_length` (`0.018006`)
6. `images` (`0.012441`)
7. `rating` (`0.012402`)
8. `image_bucket` (`0.011439`)
9. `char_per_word` (`0.004206`)
10. `rating_extremity` (`0.003117`)

Interpretation:
- **Length and structure features dominate**, consistent with helpfulness behavior in many review platforms.
- Product/review context (`log_popularity`, temporal recency proxy) has meaningful secondary influence.
- Punctuation/extremity features are weaker in this run, implying limited incremental signal once stronger structural features are present.

## 7) Reliability and Limitations (Critical)

- Some parquet variants may contain pre-engineered columns that are sparse or constant for subsets.
- Constant/low-variance features can create NaN correlations and weak visual diagnostics.
- SHAP outputs can vary by version and binary output format (2D vs 3D); robust shape handling is required.
- LIME is local and perturbation-based:
  - sensitive to random seed and neighborhood sampling,
  - should be reported with caution and consistency checks.
- This pipeline predicts an operational label (`is_helpful`), not intrinsic review quality; potential exposure to platform/process bias.

## 8) Practical Interpretation for Decision-Making

- Current model is suitable as a **ranking/routing signal** rather than a hard gate.
- AUC `0.72` indicates useful discrimination but not enough for high-stakes automation alone.
- Best use cases:
  - prioritize reviews for moderation/highlight,
  - sort candidate helpful reviews,
  - support downstream human-in-the-loop triage.

## 9) Recommended Improvements (Priority Ordered)

1. **Data quality audit first**
   - verify non-empty text ratio, feature variance, and label consistency per shard.
2. **Calibration + threshold tuning**
   - optimize for business objective (precision@k, recall floor, cost-sensitive F1).
3. **Temporal and product-group validation**
   - test drift robustness and leakage risk.
4. **More robust text encoder**
   - compare against transformer fine-tuning for expected lift.
5. **Explainability governance**
   - add multi-sample LIME stability checks,
   - report SHAP confidence intervals via bootstrapping.
6. **Self-attention over LSTM outputs** — let the model weight the most informative tokens
   instead of relying purely on final hidden states.
7. **Ensembling** — stack LSTM predictions with the TF-IDF baselines (e.g. Random Forest) for
   robustness.
8. **Sub-word tokenization** (BPE / SentencePiece) to handle out-of-vocabulary words better
   than the current fixed 50K-token vocabulary.

## 10) Reproducibility Checklist

- Run notebook in order from data loading through explainability sections.
- Confirm dataset path printed in notebook is:
  - `data/processed/amazon_reviews_s10.parquet`
- Confirm target source column printout (`is_helpful` vs vote-based fallback).
- Re-generate comparison table and explainability outputs before reporting.

## 11) Executive Conclusion

- The notebook demonstrates a meaningful end-to-end helpfulness pipeline.
- The **Hybrid BiLSTM is currently the best-performing model** in this experiment.
- Explainability outputs are coherent and support the quantitative findings.
- The most critical next step is **data robustness/consistency validation**, followed by threshold calibration and stronger encoder benchmarking.

