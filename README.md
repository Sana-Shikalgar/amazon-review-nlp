## What I've done (Sentiment & Emotion Analysis)

- Created branch feat/sentiment-analysis
- Built an inference pipeline in notebook/sentiment_analysis.ipynb using two pre-trained Hugging Face models:
  - cardiffnlp/twitter-roberta-base-sentiment-latest for positive / neutral / negative
  - j-hartmann/emotion-english-distilroberta-base for joy, anger, sadness, fear, disgust, surprise, and neutral
- Applied filtering to select the top 5 reviews per product, ranked by helpful votes with review length as a tiebreaker
- Ran inference on 10,000 reviews as a CPU development run
- Saved results to data/processed/sentiment_emotion_results.parquet
- Stored class probability scores from the models and exposed the top-1 label as the main output

## Basic checks performed

- Checked alignment between sentiment predictions and star ratings
- Reviewed model confidence distributions
- Examined emotion label distributions across the dataset
- Performed manual spot-checks on a small number of reviews