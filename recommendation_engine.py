"""
AI-Powered Product Recommendation Engine
-----------------------------------------
A small content-based recommendation system built with Python and data analytics.

Pipeline:
1. Load product catalog (sample e-commerce dataset)
2. Run exploratory data analytics on the catalog
3. Build a content-based recommender using TF-IDF + cosine similarity
   on product descriptions/categories
4. Generate top-N recommendations for a given product
5. Output analytics summary + recommendation results
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Sample product catalog (20 products across categories)
# ---------------------------------------------------------
data = {
    "product_id": range(1, 21),
    "name": [
        "Wireless Bluetooth Headphones", "Noise Cancelling Earbuds", "Over-Ear Studio Headphones",
        "Running Shoes - Lightweight", "Trail Running Shoes", "Casual Sneakers",
        "Stainless Steel Water Bottle", "Insulated Travel Mug", "Glass Water Bottle",
        "Yoga Mat - Non Slip", "Resistance Bands Set", "Adjustable Dumbbells",
        "Smart Fitness Watch", "Basic Fitness Tracker", "GPS Running Watch",
        "Organic Green Tea", "Herbal Detox Tea", "Cold Brew Coffee Bags",
        "Backpack - Laptop Compartment", "Travel Duffel Bag",
    ],
    "category": [
        "Audio", "Audio", "Audio",
        "Footwear", "Footwear", "Footwear",
        "Drinkware", "Drinkware", "Drinkware",
        "Fitness", "Fitness", "Fitness",
        "Wearables", "Wearables", "Wearables",
        "Beverages", "Beverages", "Beverages",
        "Bags", "Bags",
    ],
    "description": [
        "wireless bluetooth headphones with long battery life for music and calls",
        "compact noise cancelling earbuds for workouts and commuting",
        "over-ear studio headphones for professional audio monitoring",
        "lightweight running shoes with breathable mesh for daily runs",
        "durable trail running shoes with grip sole for outdoor terrain",
        "casual everyday sneakers with cushioned comfort sole",
        "stainless steel insulated water bottle keeps drinks cold for hours",
        "insulated travel mug keeps coffee hot during commute",
        "eco friendly glass water bottle with silicone sleeve",
        "non slip yoga mat for home workouts and stretching",
        "resistance bands set for strength training and mobility",
        "adjustable dumbbells for home gym strength workouts",
        "smart fitness watch tracks heart rate steps and workouts",
        "basic fitness tracker for step count and sleep tracking",
        "gps running watch tracks pace distance and route for runners",
        "organic green tea bags rich in antioxidants",
        "herbal detox tea blend for daily wellness routine",
        "cold brew coffee bags for smooth iced coffee at home",
        "backpack with padded laptop compartment for commuting to work",
        "spacious travel duffel bag for weekend trips and gym",
    ],
    "price": [59, 45, 89, 65, 72, 40, 22, 28, 18, 25, 20, 95, 129, 35, 149,
              12, 14, 16, 55, 48],
    "rating": [4.5, 4.3, 4.7, 4.2, 4.4, 4.0, 4.6, 4.1, 4.3, 4.5, 4.2, 4.6,
               4.4, 3.9, 4.7, 4.3, 4.1, 4.4, 4.5, 4.2],
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2. Data analytics on the catalog
# ---------------------------------------------------------
print("=== CATALOG ANALYTICS ===")
category_summary = df.groupby("category").agg(
    avg_price=("price", "mean"),
    avg_rating=("rating", "mean"),
    product_count=("product_id", "count"),
).round(2)
print(category_summary, "\n")

top_rated = df.sort_values("rating", ascending=False).head(5)[["name", "category", "rating"]]
print("Top 5 rated products:")
print(top_rated.to_string(index=False), "\n")

# ---------------------------------------------------------
# 3. Content-based recommendation engine
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])
similarity_matrix = cosine_similarity(tfidf_matrix)


def recommend(product_name, top_n=3):
    idx = df.index[df["name"] == product_name][0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    return df.iloc[[s[0] for s in scores]][["name", "category", "rating"]].assign(
        similarity=[round(s[1], 3) for s in scores]
    )


# ---------------------------------------------------------
# 4. Example: recommend products similar to a GPS running watch
# ---------------------------------------------------------
target_product = "GPS Running Watch"
print(f"=== RECOMMENDATIONS FOR: {target_product} ===")
recommendations = recommend(target_product, top_n=3)
print(recommendations.to_string(index=False))

# Save outputs for the case study
category_summary.to_csv("category_analytics.csv")
recommendations.to_csv("sample_recommendations.csv", index=False)
