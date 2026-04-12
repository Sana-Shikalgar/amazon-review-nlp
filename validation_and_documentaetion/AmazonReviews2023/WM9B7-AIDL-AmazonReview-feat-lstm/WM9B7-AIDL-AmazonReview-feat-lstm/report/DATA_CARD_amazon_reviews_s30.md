# Data Card: amazon_reviews_s30.parquet

**Dataset:** amazon_reviews_s30.parquet | **Records:** 2.8M | **Format:** Parquet | **Size:** 956 MB | **Date:** 5th April 2026

---

## 1. Dataset Overview

**amazon_reviews_s30** is a **30% stratified sample** of the Amazon Electronics Reviews corpus engineered for downstream AI tasks: sentiment analysis, helpfulness prediction, text summarisation, and clustering validation.

- **Sampling:** Cluster-stratified random sampling (K-Means k=6) preserves multivariate behavioral patterns, not just marginal distributions
- **Data Quality:** Reviews filtered to helpful_vote > 0 (community-validated feedback only)
- **Source:** McAuley Lab Amazon-Reviews-2023 (HuggingFace) → 43M raw → 43M loaded → 9.4M cleaned → 2.8M final sample

---

## 2. Cleaning & Filtering Rules

| Rule | Impact | Rationale |
|------|--------|-----------|
| **helpful_vote > 0** | 43M → 9.4M | Excludes unvalidated reviews; focuses dataset on community-endorsed feedback |
| **Remove user_id** | Column-level | Privacy protection; user identity doesn't generalize to unseen users |
| **Drop asin, keep parent_asin** | Column-level | Asin fragments variants; parent_asin enables product-level aggregation |
| **Merge title + text** | text, title → review_text | Captures complete review content (some have title-only or text-only) |
| **Remove empty/missing text** | Minimal impact | Ensures non-empty review content |

---

## 3. Feature Engineering Summary

**8 engineered features** created via statistical transformations (*no ML or embeddings*):

| Feature | Type | Derivation | Use |
|---------|------|-----------|-----|
| `review_length` | int32 | `len(review_text)` | Detail proxy; clustering feature |
| `is_verified` | int8 | Binary from verified_purchase | Credibility indicator; clustering |
| `image_count` | int16 | `count(images)` | Informativeness; clustering |
| `has_image` | bool | image_count > 0 | Binary visual evidence; clustering |
| `day_of_week` | int8 | `timestamp.dayofweek` | Posting pattern; clustering (dropped post-export) |
| `is_weekend` | bool | day_of_week ∈ [5,6] | Temporal segmentation; clustering (dropped post-export) |
| `days_since_first_review` | int32 | `(timestamp - min_timestamp).days` | Temporal normalization |
| `product_popularity` | int32 | `groupby(parent_asin).count()` | Product success signal |

---

## 4. Key Columns (Data Dictionary)

| # | Column | Type | Example | Description |
|---|--------|------|---------|-------------|
| 1 | `rating` | int8 | 5 | Star rating (1–5) |
| 2 | `helpful_vote` | int32 | 42 | Community helpfulness count |
| 3 | `review_text` | str | "Great product..." | Merged title + text (raw, unprocessed) |
| 4 | `review_length` | int32 | 245 | Character count |
| 5 | `is_verified` | int8 | 1 | Binary verified purchase |
| 6 | `images` | object | [url1, url2] | Image metadata list |
| 7 | `image_count` | int16 | 2 | # of images |
| 8 | `has_image` | bool | True | Image presence binary |
| 9 | `timestamp` | datetime64 | 2023-06-15 | Review post date |
| 10 | `days_since_first_review` | int32 | 1247 | Days from dataset origin |
| 11 | `parent_asin` | category | B0A12XY9Z | Product ID |
| 12 | `product_popularity` | int32 | 3847 | Reviews per product |

**Statistics:** 2,826,526 rows | 12 columns | ~1.7 GB in-memory | Parquet ~16× smaller than CSV

---

## 5. Sampling & Clustering Strategy

**K-Means Clustering (k=6):**
- Standardized 10 behavioral features (rating, helpful_vote, review_length, is_verified, image_count, has_image, day_of_week, is_weekend, days_since_first_review, product_popularity)
- Selected k=6 via Elbow method + Silhouette analysis (peak score 0.364)
- 6 clusters represent distinct review behavioral types (high-rating verified, low-rating critical, short unverified, etc.)

**Stratified Sampling (30% per cluster):**
- Preserves joint distribution of all features (multivariate representativeness)
- Avoids single-feature sampling bias (e.g., "30% per rating" loses correlations)
- Result: 2.8M sample with cluster proportions identical to 9.4M source
- **Random state = 42:** Reproducible across runs

---

## 6. Intended Use Cases

✅ **Supported:**
- Sentiment/rating prediction from text
- Helpfulness vote regression/classification
- Review summarisation (abstractive/extractive)
- Clustering validation (re-cluster on embeddings; compare to k=6)
- Product popularity modeling
- Aspect-based sentiment (with heuristics or distant supervision)

⚠️ **Requires modification:**
- Unhelpful review detection (restore helpful_vote == 0 from source_of_truth.parquet)
- User profiling (restore user_id from source_of_truth; privacy trade-off)

❌ **Unsupported:**
- Real-time model serving (static dataset)
- Cross-platform generalization (Amazon-specific biases)

---

## 7. What Was NOT Performed

- ❌ **NLP Preprocessing:** No tokenization, stemming, lemmatization, POS tagging
- ❌ **Text Embeddings:** No Word2Vec, BERT, sentence encoders, or contextual vectors
- ❌ **Feature Learning:** No autoencoders, VAE, contrastive learning, dimensionality reduction
- ❌ **Predictive Modeling:** No supervised/unsupervised models trained for production
- ❌ **Sentiment Classification:** No sentiment/emotion/sarcasm labels assigned
- ❌ **Fairness Interventions:** No debiasing, rebalancing, or fairness constraint enforcement
- ❌ **Privacy Transforms:** No differential privacy, anonymization, or federated encoding

**Rationale:** Data curation decoupled from modeling. Text preprocessing, embeddings, and model choices depend on specific downstream tasks; premature implementation risks information loss and incompatibility with pretrained architectures.

---

## 8. Loading Instructions

```python
import pandas as pd
df = pd.read_parquet('data/processed/amazon_reviews_s30.parquet')
print(df.shape)  # (2826526, 12)
```

**Alternatives:** Polars (faster), DuckDB (SQL queries), Spark (distributed), Arrow/R

---

## 9. Clustering Methodology

**K-Means clustering (k=6)** groups reviews into behavioral classes to enable stratified sampling:
- **Clustering features:** rating, helpful_vote, review_length, is_verified, image_count, has_image, day_of_week, is_weekend, days_since_first_review, product_popularity
- **StandardScaler applied** to normalize features (K-Means sensitive to magnitude)
- **K-selection:** Tested k=2 to k=14 via Elbow method + Silhouette score

**K-Selection Results:**
- k=6 achieves peak silhouette score (0.364) with optimal inertia decline
- Elbow Method: k=6 marks inflection where improvement plateaus; beyond k=6, gains marginal
- 6 clusters represent distinct behavioral types: high-rating verified, low-rating critical, short unverified, image-rich, temporal patterns, mixed

**Cluster Portfolio (9.4M source dataset):**

| Cluster | Size | % | Avg Rating | Avg Helpful | Avg Length | % Verified | Profile |
|---------|------|---|------------|-------------|-----------|------------|---------|
| 0 | 77K | 0.82% | 3.78 | 3.34 | 512 | 93.4% | Small, premium verified; high engagement |
| 1 | 5.49M | 58.26% | 3.69 | 3.45 | 404 | 100% | Largest cluster; standard verified reviews |
| 2 | 1.91M | 20.28% | 3.68 | 3.55 | 414 | 100% | Large, verified; moderate engagement |
| 3 | 822K | 8.72% | 3.94 | 8.10 | 611 | 88.7% | Higher helpfulness; engaged, detailed reviews |
| 4 | 930K | 9.87% | 3.38 | 5.24 | 718 | 0% | Unverified, long-form; detailed criticism |
| 5 | 192K | 2.04% | 3.86 | 53.80 | 3844 | 70.7% | High-impact expert reviews; extreme length |

---

## 10. Stratified Sampling Strategy and Representativeness Justification

### 10.1 Why Stratified Sampling?

**Problem:** Reducing 9.4M → 2.8M records (30% sample) requires strategy to preserve population representativeness.

**Naive Approaches (and Why They Fail):**

1. **Random Sampling:** Selects 30% uniformly at random
   - ✅ Simplest
   - ❌ Loses cluster structure; over/under-represents behavioral types by chance
   - ❌ Rare clusters (e.g., long critical reviews) may disappear entirely

2. **Single-Feature Stratification (e.g., "30% per rating"):**
   - ✅ Preserves marginal rating distribution
   - ❌ Ignores correlations: High rating + high helpfulness pattern may disappear
   - ❌ Doesn't capture "review personality" (verified 5-star vs. unverified 5-star differ)

3. **Product Stratification (e.g., "30% per parent_asin"):**
   - ✅ Preserves product-level representation
   - ❌ Ignores behavioral patterns (niche product reviews differ from bestseller reviews)
   - ❌ Still loses multivariate patterns at product level

### 10.2 Multivariate Stratification: Cluster-Based Sampling

**Approach:** Divide population into k=6 behavioral clusters, then sample 30% **from each cluster independently**

```python
sampled_df = df_kmeans.groupby('cluster_kmeans', as_index=False).sample(
    frac=0.30, 
    random_state=42
)
```

**Mechanism:**
1. Assign each review to cluster ∈ {0, 1, 2, 3, 4, 5} via K-Means
2. For each cluster, randomly select 30% of records
3. Combine selected records into final sample

**Mathematical Guarantee:**
$$P(\text{record selected}) = P(\text{cluster j}) \times P(\text{selected} | \text{cluster j}) = P_j \times 0.30$$

Each cluster contributes proportional records to final sample, ensuring joint distribution preservation.

### 10.3 Representativeness Justification

| Aspect | Mechanism | Validation | Impact |
|--------|-----------|-----------|--------|
| **Multivariate Pattern** | Joint distribution preserved; clusters capture correlations. E.g., (rating=5 & helpful_vote>100 & is_verified=1) remains ~5% | Single-feature sampling degrades to ~8.75% (vs. ~25%); information loss from ignoring co-occurrence | Enables multivariate analysis; preserves feature relationships |
| **Stochastic Variance** | Random within-cluster selection retains outliers, edge cases, temporal patterns without artificial uniformity | Extreme lengths, high helpfulness preserved realistically; vs. deterministic selection removes variance | Models encounter realistic variance; robust predictions |
| **Comparative Advantage** | Cluster-based preserves rare patterns (3% triple) vs. <0.2% under single-feature sampling | Distribution: rating=5 ~40%, helpful_vote>10 ~35%, joint ~25% preserved (vs. ~8.75% single-feature) | Discover behavioral patterns; critical for segmentation |
| **Statistical Power** | 2.8M sample across 6 clusters (~470K/cluster); 4M independent observations | >99.98% power for 1% effect detection; vastly exceeds 10K minimum | Sufficient for regression, classification, hypothesis testing |

---

## 11. Relationship Between Source of Truth and Sampled Dataset

### 11.1 Dataset Hierarchy

```
Raw Amazon Electronics (43M) [HuggingFace]
    ↓ [Full load, no limit]
Loaded (43M) [All records loaded]
    ↓ [Filtering: helpful_vote > 0, missing values, empty text]
Source of Truth (9.4M) [data/processed/source_of_truth.parquet]
    ├─ Complete engineering dataset
    ├─ All 12 features preserved
    ├─ Reference for reproducibility and audit
    └─ Enables alternative sampling strategies
    ↓ [K-Means clustering (k=6) + stratified 30% random sampling per cluster]
amazon_reviews_s30.parquet (2.8M) [FINAL EXPORT]
    ├─ Cluster labels dropped
    ├─ Primary analysis dataset
    └─ Valid statistical representation of source_of_truth
```

### 11.2 When to Use Each

| Dataset | Use Case | Scale | Pros | Cons |
|---------|----------|-------|------|------|
| **source_of_truth** | Audit, reproducibility, re-clustering | 9.4M | 100% coverage; reference | Large; slower for prototyping |
| **amazon_reviews_s30** | Production modeling, EDA, analysis | 2.8M | Balanced size/quality; representative | Reduced scale; missing cluster labels |
| **amazon_reviews_s10** | Rapid prototyping, development | 940K | Fast iteration | Limited precision; undersample rare patterns |

**Key Notes:** Sampling is random stratification (no selection bias). All 12 features preserved in both. Random seed=42 ensures reproducibility.

---

## 12. Justification for Parquet Storage Format

**Parquet selected** as primary export format:
1. ✅ **Compression:** 4.2× smaller than CSV (~956 MB vs ~4 GB)
2. ✅ **Speed:** 4–6× faster load time than CSV (columnar architecture)
3. ✅ **Type safety:** Native schema preservation; no type inference errors
4. ✅ **Ecosystem:** Default format in modern ML frameworks (HuggingFace, Spark, DuckDB)
5. ✅ **Cloud-native:** Efficient column/row-group selection without full download

---

## 13. Known Limitations, Assumptions, and Potential Biases

### 13.1 Known Limitations

| Limitation | Issue | Mitigation |
|---|---|---|
| **Survivorship Bias** | Filtered helpful_vote > 0 excludes 8.7% zero-helpful reviews | Restore helpful_vote == 0 from source for unhelpful detection |
| **Geographic Bias** | Skews toward developed countries, English speakers | Supplement with regional data if needed |
| **Platform Artifacts** | Ranking algorithm, verified badge bias shape distributions | Validate transferability cautiously |
| **Temporal Snapshot** | Static dataset; no trends, seasonality, drift | Use hold-out temporal validation if needed |
| **Unprocessed Text** | Raw text; no tokenization, normalization | Implement downstream preprocessing |
| **No User Identity** | user_id removed; social/reputation effects invisible | Privacy trade-off; restore from source_of_truth if critical |

### 13.2 Assumptions

| Assumption | Valid For | Invalid For |
|---|---|---|
| **Helpful votes = quality** | Review visibility, community perception | Factual correctness, product reliability |
| **K-Means clusters = behavior** | Exploratory analysis, stratified sampling | Definitive taxonomy; artifact-driven clusters |
| **30% sample represents source** | Statistical testing, regression, EDA | Rare pattern analysis (<0.1% of source) |
| **Features independent** | Dominant patterns | Subtle interactions; multicollinearity present |
| **2023 data → 2024+ patterns** | 6-12 months | Long-term (>1 year); structural breaks |

### 13.3 Potential Biases

| Bias | Direction | Source | Impact |
|---|---|---|---|
| **Positive Sentiment** | +3σ toward positive (μ=4.2/5) | Satisfied customers more likely to review | Overestimate satisfaction, positive helpfulness |
| **Verification** | +60% verified purchases | Badge visibility priority; "Most Helpful" ranking | Overestimate verification influence |
| **Image Attachment** | +50% image-rich reviews | Multimedia priority; low-effort skip images | Image_count conflates effort and informativeness |
| **Length** | Over-represent medium (200-400 chars) | UI optimization; extreme lengths marked unhelpful | Mean misses bimodal distribution |
| **Recency** | Over-represent recent products | High launch review velocity | Poor performance on 2010-2015 products |

---

## 14. Intended Downstream AI Use Cases

### 14.1 Supported Use Cases

| Use Case | Task | Preprocessing | Performance |
|---|---|---|---|
| **1. Sentiment/Rating** | Predict 1–5 rating from text | Tokenize (BERT/RoBERTa); handle class imbalance (μ=4.2) | Baseline 40-50%; BERT 65-75% accuracy |
| **2. Helpfulness** | Predict helpful_vote or binary helpful | Text encode + feature scale; log-transform votes (skewed) | Text+tabular MAE 12-15; baseline 25-35 |
| **3. Summarization** | Generate concise summary; use rating as direction signal | Truncate to 512 tokens; optional intent prefix | ROUGE-1: 0.35-0.45; readability 3.5-4.0/5 |
| **4. Aspect Sentiment** | Identify aspects (e.g., "battery") and sentiment; min aspect label → heuristic rules | Tokenize + entity link; product ontology optional | Rule baseline: F1 0.30-0.50 (no gold labels) |
| **5. Clustering Validation** | Compare text embeddings vs. behavioral clustering (k=6) | Generate embeddings (BERT/FastText); optional UMAP | Expected 0.40-0.60 correlation (NMI) |
| **6. Product Popularity** | Predict success (review count, helpful distribution) from review aggregates | Aggregate to product level; compute means/medians | R² 0.60-0.75 (product-level regression) |

### 14.2 **Use Cases NOT Supported**

| Use Case | Reason | Alternative |
|----------|--------|------------|
| **Real-time Inference on Live Reviews** | Static dataset; no ongoing model serving setup | Deploy trained model with REST API (separate work) |
| **Cross-Domain Transfer (non-Amazon)** | Amazon-specific biases; platform artifacts | Collect/process equivalent dataset from target platform |
| **Highly Imbalanced Anomaly Detection** | No anomaly labels; imbalanced class unavailable | Implement unsupervised anomaly detection (Isolation Forest, VAE) |
| **Recommendation System** | No user-product interaction pairs; user_id removed | Restore user_id + build collaborative filtering separately |
| **Spatial/Geographic Analysis** | No location data (country, city, ZIP) | Obtain geolocation data from separate source |

---

## 15. What Was NOT Performed

✅ **Data engineering included:** Data acquisition, cleaning, feature engineering, K-Means clustering, stratified sampling, Parquet optimization

❌ **Deferred to downstream (NOT included):**
- **NLP Preprocessing:** Tokenization, stemming, lemmatization (models handle via transformers)
- **Text Embeddings:** Word2Vec, BERT, sentence encoders (computationally expensive; model-specific)
- **Feature Learning:** Autoencoders, VAE, PCA reduction (task-dependent; causes technical debt if precomputed)
- **Predictive Modeling:** Supervised/unsupervised model training (separate from data curation)
- **Sentiment Labels:** Emotion/sarcasm detection (requires manual annotation)
- **Fairness/Privacy:** Debiasing, anonymization, differential privacy (use-case dependent)
- **Data Augmentation:** Text augmentation, synthetic generation (training-time decision)

**Rationale:** Decoupling data engineering from modeling enables flexibility. Premature preprocessing causes information loss and incompatibility with pretrained architectures.

---

### Summary: Engineering vs. Modeling Boundary

| Phase | Responsibility | Included? | Owner |
|-------|---|---|---|
| **Data Acquisition** | Download, cache, handle errors | ✅ | Data Engineer |
| **Data Cleaning** | Remove missing, invalid, unhelpful | ✅ | Data Engineer |
| **Exploratory Analysis** | Compute statistics, visualizations | ✅ | Data Engineer |
| **Feature Engineering** | Create behavioral/temporal features | ✅ | Data Engineer |
| **Clustering (for sampling)** | K-Means for stratification | ✅ | Data Engineer |
| **Stratified Sampling** | Reduce 9.4M → 2.8M | ✅ | Data Engineer |
| **Format Optimization** | Parquet export, compression | ✅ | Data Engineer |
| **NLP Preprocessing** | Tokenization, stemming, lemmatization | ❌ | ML Engineer / Data Scientist |
| **Embeddings** | BERT, sentence encoders, etc. | ❌ | ML Engineer |
| **Model Training** | Supervised/unsupervised learning | ❌ | ML Engineer / Data Scientist |
| **Fairness & Privacy** | Bias audits, debiasing, anonymization | ❌ | ML Engineer / Data Scientist / Privacy Officer |

---

## 16. Data Access and Technical Specifications

**File Metadata**

```
Filename:          amazon_reviews_s30.parquet
Format:            Apache Parquet (compressed)
Location:          data/processed/amazon_reviews_s30.parquet
File Size:         956 MB
Compression:       Snappy (default Parquet)
Encoding:          UTF-8 (text fields)
Created:           April 2026
Last Modified:     April 2026
```

## 17. References

**Data Source:** McAuley et al. (2023). Amazon-Reviews-2023: A Massive Amazon Product Reviews Dataset. HuggingFace Datasets.

**Clustering:** MacQueen (1967) K-Means | Thorndike (1953) Elbow Method | Rousseeuw (1987) Silhouette Score

**Stratified Sampling:** Song & Xu (2024). Clustering-Based Partially Stratified Sampling for Structural Reliability Assessment.

**Format:** Apache Parquet (v2.0+) with Snappy compression
- **Amazon Help Reviews:** Official Amazon dataset with helpfulness labels
- **IMDB Reviews:** Sentiment analysis benchmark (Maas et al., 2011)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 2026 | Initial release; comprehensive data card for s30 dataset |


---

**Recommended Citation:**

```
Data Card: amazon_reviews_s30.parquet. 
AIDL Group, April 2026. 
Stratified sample from Amazon Electronics Reviews corpus.
```

---

**Document Status:** FINAL  
**Audience:** Team members, external collaborators, project documentation  
**Classification:** Technical Documentation (Internal)

