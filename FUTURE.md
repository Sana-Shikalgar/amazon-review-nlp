# Future Work

Gaps and improvements identified while consolidating this repo (branch merges, environment
setup, and full end-to-end execution of every notebook), plus recommendations carried over
from the individual model reports.

## Data pipeline

- **Train/val/test split is per-model, not shared.** Each model notebook
  (`06_lstm.ipynb`, `07_bert.ipynb`, `08_multimodel.ipynb`) performs its own split directly from
  `s03_filter.parquet` / `multimodal_s03_filter.parquet` (see `notebook/train_test_split_protocol.ipynb`
  for the rationale) — this let each model be developed and iterated on independently. Moving to
  one shared split would make cross-model comparisons more direct.
- **Image downloader reliability.** `04_image_downloader.ipynb` has no retry/backoff and takes
  ~90 minutes end-to-end (73K images) with a nontrivial timeout/connection-error rate. Worth
  adding retries with backoff, or caching partial results more robustly.

## Modeling

- **Checkpoints exist locally but aren't version-controlled.** `08_multimodel.ipynb` saves a
  real checkpoint to `models/` (`multimodal_model_full.pt`, `multimodal_model_metadata_full.pkl`),
  but `*.pt`/`*.h5`/`*.hdf5` are gitignored (too large), so LSTM and BERT must still be
  retrained from scratch to reproduce results. Consider a lightweight metrics-JSON artifact
  store per run so results don't have to be re-derived by reading notebook cell outputs.
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
