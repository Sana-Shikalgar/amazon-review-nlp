# Future Work

Gaps and improvements identified while consolidating this repo (branch merges, environment
setup, and full end-to-end execution of every notebook), plus recommendations carried over
from the individual model reports.

## Data pipeline

- **Feature-selection guidance is contradictory across reports.** `reports/data/Final_Feature_Selection_Report.md`
  recommends keeping `word_count` and dropping `review_length` (and keeping `has_image` over
  `image_count`); `reports/data/Feature_Redundancy_Analysis.docx`'s quantitative VIF/correlation
  analysis recommends the opposite on both (keep `review_length` over `word_count`, r=0.997/VIF=247.7;
  keep `image_count` over `has_image`). These need to be reconciled — the docx's analysis is backed
  by explicit VIF numbers and is the more likely one to trust, but the team should confirm which
  feature set `02_feature_engineering.ipynb` should actually standardize on.
- **`amazon_reviews_s30.parquet` has no reproducing notebook.** The current `notebook/01`–`03`
  pipeline only reproduces `amazon_reviews_s10.parquet` (a 10% cluster-stratified sample).
  `s30` (the 30% sample most models and `05_visualization.ipynb` actually load) was produced
  by an earlier/since-renamed version of this pipeline. Either regenerate `03_clustering_sampling.ipynb`
  with a documented `SAMPLED_FRACTION` for `s30`, or retire `s30` in favor of `s10` everywhere
  and update the notebooks/docs that reference it.
- **No consistent train/val/test split shared across models.** `05_visualization.ipynb` exports
  a temporal split (`data/splits/train.parquet`, `val.parquet`, `test.parquet`); the model notebooks
  each do their own splitting instead of consuming it. Standardizing on one split would make
  results genuinely comparable across models.
- **Image downloader reliability.** `04_image_downloader.ipynb` has no retry/backoff and takes
  ~90 minutes end-to-end (73K images) with a nontrivial timeout/connection-error rate. Worth
  adding retries with backoff, or caching partial results more robustly.

## Modeling

- **No saved model checkpoints or artifacts anywhere.** Every model (LSTM, BERT, multimodal,
  sentiment/emotion) must be fully retrained from scratch to reproduce results — there's no
  `models/*.pt` or metrics file checked in (by design, `*.pt`/`*.h5`/`*.hdf5` are gitignored).
  Consider a lightweight artifact store (even just checked-in metrics JSON per run) so results
  don't have to be re-derived by reading notebook cell outputs.
- **Not apples-to-apples.** LSTM trains on 846K rows, BERT on a 48K-review subset, the
  multimodal model on ~73K rows with images. A shared benchmark harness (same split, same
  eval script) would make "best model" comparisons meaningful rather than approximate.
- **`models/` folder name collision.** The multimodal notebook saves its checkpoint to
  `../models/multimodal_model_full.pt`, which (after this restructuring) resolves inside the
  `models/` notebooks folder rather than a dedicated checkpoints directory. Rename one of the
  two, e.g. a `checkpoints/` folder for saved weights, separate from `models/` (notebooks).
- **More robust text encoder for helpfulness prediction** — compare the Hybrid BiLSTM against
  a fine-tuned transformer on the same full-size sample the LSTM uses (846K rows), not just
  BERT's smaller 48K-review subset.
- **Calibration + threshold tuning** for the helpfulness classifiers, optimized for the actual
  downstream use case (precision@k, recall floor, or cost-sensitive F1) rather than raw accuracy.
- **Temporal and product-group validation** to check drift robustness and leakage risk beyond
  a single random split.

## Sentiment & emotion analysis

- **Fear label is unreliable** — bimodal across ratings 1 and 5 (43% positive-sentiment overlap)
  per the notebook's own critical analysis; exclude it from downstream use or replace the
  emotion model.
- **`[SEP]` token and 512-token truncation bias** — the sentiment model is out-of-distribution
  for the `[SEP]`-joined title+body field, and long reviews get truncated to their opening
  section rather than their conclusion. Replacing `[SEP]` with a neutral separator and/or
  windowing long reviews (last 512 tokens, or a summarization pass) would reduce this bias.
- **3-star boundary miscalibration** — the sentiment model over-predicts negative on ambivalent
  3-star reviews (38.2% negative vs. the 33% random baseline for 3 classes). Worth a dedicated
  calibration pass on the boundary classes.

## Explainability

- **LIME stability** — add multi-sample LIME runs to check explanation stability rather than a
  single perturbation run per reviewed example.
- **SHAP confidence intervals** — bootstrap SHAP attributions to report confidence intervals
  rather than point estimates.

## Engineering / process

- **No CI or automated notebook execution testing.** This consolidation's "does everything run"
  check was a manual, one-time pass. A scheduled or PR-triggered `nbconvert --execute` run
  (even just on the fast data notebooks) would catch path/dependency breakage automatically.
- **Avoid blocking `input()` calls and hardcoded sample-size assumptions in notebooks** — both
  were found and fixed during this consolidation (a blocking prompt for k in the clustering
  notebook, and silhouette/visualization sampling that assumed ≥100K rows). Non-interactive
  execution should be a review criterion for new notebook cells.
- **Git LFS tracking must be reconciled deliberately when merging branches** — several source
  branches had `.gitattributes` silently emptied at some point, which would have unpinned
  `*.parquet`/`*.csv` from LFS tracking if not caught during this merge.
