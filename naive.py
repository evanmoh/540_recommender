import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the coffee shop data
coffee_shops = pd.read_csv('coffeeshop_data_cleaned.csv')

# Extract ratings to test against
ratings_data = []
for i in range(len(coffee_shops)):
    shop_id = coffee_shops.iloc[i]['id']
    for j in range(1, 11):
        review_rating = coffee_shops.iloc[i][f'review{j}rating']
        if pd.notna(review_rating):
            ratings_data.append({
                'shopId': shop_id,
                'actual_rating': review_rating
            })

ratings = pd.DataFrame(ratings_data)
print(f"Got {len(ratings)} ratings to test on")

# Use the shop's Yelp rating for predictions
def predict_yelp_rating(shop_id):
    shop_rating = coffee_shops[coffee_shops['id'] == shop_id]['rating'].iloc[0]
    return shop_rating

# Make predictions for RMSE calculation
predictions = []
for _, row in ratings.iterrows():
    pred = predict_yelp_rating(row['shopId'])
    predictions.append(pred)

rmse = np.sqrt(mean_squared_error(ratings['actual_rating'], predictions))
print(f"RMSE (Yelp rating model): {rmse:.3f}")

# Recommend highest Yelp rated shops
def recommend_by_yelp(n_recs=5):
    top_shops = coffee_shops.sort_values('rating', ascending=False).head(n_recs)
    recs = []
    for _, row in top_shops.iterrows():
        recs.append({
            'shop_name': row['name'],
            'predicted_rating': row['rating']
        })
    return recs

print("\nTop Recommendations (Yelp rating):")
yelp_recs = recommend_by_yelp()
for i, rec in enumerate(yelp_recs, 1):
    print(f"{i}. {rec['shop_name']} (predicted: {rec['predicted_rating']:.2f})")
