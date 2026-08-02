# AI-Powered Product Recommendation Engine

A content-based product recommendation system built with Python, pandas, and scikit-learn. Suggests similar products from a catalog using TF-IDF vectorization and cosine similarity — the same core idea behind "customers also viewed" features on e-commerce platforms.

**[Live Demo](#)** ← replace with your GitHub Pages link once deployed

## Problem

Recommendation systems traditionally rely on user purchase history. This fails for new products or platforms with no interaction data yet (the "cold-start problem"). This project generates relevant recommendations purely from product descriptions and categories — no purchase history required.

## How It Works

1. **Catalog** — 20 sample products across 7 categories (Audio, Footwear, Fitness, Wearables, etc.)
2. **Data Analytics** — pandas summarizes average price, average rating, and item count per category
3. **Feature Extraction** — product descriptions are vectorized using TF-IDF (scikit-learn)
4. **Similarity Scoring** — cosine similarity ranks how closely related any two products are
5. **Recommendation** — `recommend(product_name, top_n)` returns the top-N most similar products

## Example Output

Query: `GPS Running Watch`

| Recommended Product | Category | Rating | Similarity |
|---|---|---|---|
| Smart Fitness Watch | Wearables | 4.4 | 0.221 |
| Running Shoes - Lightweight | Footwear | 4.2 | 0.095 |
| Trail Running Shoes | Footwear | 4.4 | 0.088 |

## Tech Stack

- Python
- pandas
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)

## Run It Locally

```bash
pip install pandas scikit-learn
python3 recommendation_engine.py
```

## Files

- `recommendation_engine.py` — the core Python recommendation pipeline
- `index.html` — interactive browser demo (deployable via GitHub Pages)

## Skills Demonstrated

Python programming · Data cleaning & preprocessing · Exploratory data analysis · Feature engineering (TF-IDF) · Similarity-based machine learning · Turning technical output into a usable feature
