import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import re

import pickle

# Load the coffee shop data
coffee_shops = pd.read_csv('data/coffeeshop_data_cleaned.csv')

print("Loaded", len(coffee_shops), "coffee shops")

# Extract reviews with NLP analysis
reviews_data = []
for i in range(len(coffee_shops)):
    shop_id = coffee_shops.iloc[i]['id']
    shop_name = coffee_shops.iloc[i]['name']
    
    # Get all reviews for this shop
    for j in range(1, 11):
        review_text = coffee_shops.iloc[i][f'review{j}']
        review_rating = coffee_shops.iloc[i][f'review{j}rating']
        
        if pd.notna(review_text) and pd.notna(review_rating):
            reviews_data.append({
                'userId': f'user_{i}_{j}',
                'shopId': shop_id,
                'shop_name': shop_name,
                'review_text': review_text,
                'rating': review_rating
            })

reviews = pd.DataFrame(reviews_data)
print("Created", len(reviews), "reviews")

# NLP: Extract what users liked/disliked from review text
def analyze_review_sentiment(text, rating):
    """Extract features from review text based on what user mentioned"""
    text = text.lower()
    
    # Initialize feature scores
    features = {
        'coffee_quality': 0,
        'food_quality': 0,
        'service_quality': 0,
        'atmosphere': 0,
        'wifi_work': 0,
        'space_seating': 0,
        'pricing': 0,
        'location': 0
    }
    
    # Coffee mentions
    coffee_words = ['coffee', 'espresso', 'latte', 'cappuccino', 'brew', 'roast']
    if any(word in text for word in coffee_words):
        if rating >= 4:
            features['coffee_quality'] = 1
        elif rating <= 2:
            features['coffee_quality'] = -1
    
    # Food mentions
    food_words = ['food', 'breakfast', 'lunch', 'sandwich', 'pastry', 'meal']
    if any(word in text for word in food_words):
        if rating >= 4:
            features['food_quality'] = 1
        elif rating <= 2:
            features['food_quality'] = -1
    
    # Service mentions
    service_words = ['service', 'staff', 'server', 'waitress', 'friendly', 'rude']
    if any(word in text for word in service_words):
        if rating >= 4:
            features['service_quality'] = 1
        elif rating <= 2:
            features['service_quality'] = -1
    
    # Atmosphere mentions
    atmosphere_words = ['atmosphere', 'vibe', 'ambiance', 'cozy', 'loud', 'quiet']
    if any(word in text for word in atmosphere_words):
        if rating >= 4:
            features['atmosphere'] = 1
        elif rating <= 2:
            features['atmosphere'] = -1
    
    # WiFi/work mentions
    work_words = ['wifi', 'work', 'study', 'laptop', 'internet']
    if any(word in text for word in work_words):
        if rating >= 4:
            features['wifi_work'] = 1
        elif rating <= 2:
            features['wifi_work'] = -1
    
    # Space/seating mentions
    space_words = ['space', 'seating', 'crowded', 'room', 'tables']
    if any(word in text for word in space_words):
        if rating >= 4:
            features['space_seating'] = 1
        elif rating <= 2:
            features['space_seating'] = -1
    
    # Price mentions
    price_words = ['price', 'expensive', 'cheap', 'cost', 'money']
    if any(word in text for word in price_words):
        if rating >= 4:
            features['pricing'] = 1
        elif rating <= 2:
            features['pricing'] = -1
    
    # Location mentions
    location_words = ['location', 'convenient', 'parking', 'accessible']
    if any(word in text for word in location_words):
        if rating >= 4:
            features['location'] = 1
        elif rating <= 2:
            features['location'] = -1
    
    return features

# Apply NLP analysis to each review
print("Analyzing review sentiment...")
review_features = []
for idx, row in reviews.iterrows():
    features = analyze_review_sentiment(row['review_text'], row['rating'])
    review_features.append(features)

# Convert to DataFrame
review_features_df = pd.DataFrame(review_features)
review_features_df['userId'] = reviews['userId']
review_features_df['shopId'] = reviews['shopId']
review_features_df['rating'] = reviews['rating']

print("Review features extracted")
print("Sample user preferences:")
print(review_features_df.head())

# Helper to get average shop profile excluding a given review idx
def get_shop_profile(shop_id, exclude_idx=None):
    shop_reviews = review_features_df[review_features_df['shopId'] == shop_id]
    if exclude_idx is not None:
        shop_reviews = shop_reviews.drop(exclude_idx)
    aspects = shop_reviews[feature_names].mean().values if not shop_reviews.empty else np.zeros(len(feature_names))
    tfidf_vec = review_tfidf[shop_reviews.index].mean(axis=0).A1 if not shop_reviews.empty else np.zeros(review_tfidf.shape[1])
    return aspects, tfidf_vec

# Create user profiles from their review preferences
user_profiles = {}
for user_id in review_features_df['userId'].unique():
    user_data = review_features_df[review_features_df['userId'] == user_id]
    # Average their preferences across all reviews
    profile = user_data[['coffee_quality', 'food_quality', 'service_quality', 
                        'atmosphere', 'wifi_work', 'space_seating', 'pricing', 'location']].mean()
    user_profiles[user_id] = profile.to_dict()

print(f"Created profiles for {len(user_profiles)} users")

# Also create restaurant profiles from user reviews
restaurant_profiles = {}
feature_names = ['coffee_quality', 'food_quality', 'service_quality', 
                'atmosphere', 'wifi_work', 'space_seating', 'pricing', 'location']

for shop_id in review_features_df['shopId'].unique():
    shop_data = review_features_df[review_features_df['shopId'] == shop_id]
    # Average what users said about this restaurant
    profile = shop_data[feature_names].mean()
    restaurant_profiles[shop_id] = profile.to_dict()

print(f"Created restaurant profiles for {len(restaurant_profiles)} restaurants")

# Show example restaurant profile
sample_shop = list(restaurant_profiles.keys())[0]
sample_shop_name = coffee_shops[coffee_shops['id'] == sample_shop]['name'].iloc[0]
print(f"\nExample: {sample_shop_name} profile from reviews:")
for feature, score in restaurant_profiles[sample_shop].items():
    if abs(score) > 0.1:
        sentiment = "good" if score > 0 else "poor"
        print(f"  {feature}: {score:.2f} ({sentiment})")

# Also use TF-IDF on review text for additional features
tfidf = TfidfVectorizer(max_features=50, stop_words='english')
review_tfidf = tfidf.fit_transform(reviews['review_text'])

print(f"TF-IDF features: {review_tfidf.shape}")

# Combine NLP sentiment features with TF-IDF
feature_names = ['coffee_quality', 'food_quality', 'service_quality', 
                'atmosphere', 'wifi_work', 'space_seating', 'pricing', 'location']


# Build personalized feature matrix (user review + shop profile)
X_features = []
y_ratings = []

for idx, row in review_features_df.iterrows():
    # User features
    user_aspects = np.array([row[fn] for fn in feature_names])
    user_tfidf = review_tfidf[idx].toarray()[0]
    # Shop profile
    shop_aspects, shop_tfidf = get_shop_profile(row['shopId'], exclude_idx=idx)
    # Combine
    features = np.concatenate([user_aspects, user_tfidf, shop_aspects, shop_tfidf])
    X_features.append(features)
    y_ratings.append(row['rating'])

X_features = np.array(X_features)
y_ratings = np.array(y_ratings)
print("Personalized feature matrix shape:", X_features.shape)


# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X_features, y_ratings, test_size=0.2, random_state=42)

# Also split the review data
review_train, review_test = train_test_split(review_features_df, test_size=0.2, random_state=42)

#ML model
ridge = Ridge()
ridge.fit(X_train, y_train)


ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
print(f"Ridge Regression RMSE: {ridge_rmse:.3f}")

shop_profiles = {}
for shop_id in coffee_shops['id']:
    aspects, tfidf_vec = get_shop_profile(shop_id)
    shop_profiles[shop_id] = (aspects, tfidf_vec)

with open('shop_profiles.pkl', 'wb') as f:
    pickle.dump(shop_profiles, f)



def recommend_shops_ridge(user_id, top_n=5):
    # Get user’s average aspects from their previous reviews
    user_reviews = reviews[reviews['userId'] == user_id]
    user_aspects = user_profiles[user_id]  # This is a dict
    user_aspects_vec = np.array([user_aspects[f] for f in feature_names])
    user_tfidf = review_tfidf[user_reviews.index].mean(axis=0).A1 if not user_reviews.empty else np.zeros(review_tfidf.shape[1])
    visited_shops = set(user_reviews['shopId'])
    recs = []
    for shop_id in coffee_shops['id']:
        if shop_id not in visited_shops:
            # Shop features
            shop_aspects, shop_tfidf = get_shop_profile(shop_id)
            # Combine all
            feature_vector = np.concatenate([user_aspects_vec, user_tfidf, shop_aspects, shop_tfidf])
            predicted_rating = ridge.predict([feature_vector])[0]
            shop_name = coffee_shops[coffee_shops['id'] == shop_id]['name'].iloc[0]
            recs.append({'shop_id': shop_id, 'shop_name': shop_name, 'predicted_rating': predicted_rating})
    recs = sorted(recs, key=lambda x: x['predicted_rating'], reverse=True)
    return recs[:top_n]


test_user = reviews['userId'].iloc[0]
recs = recommend_shops_ridge(test_user, top_n=5)
print(f"Top recommendations for {test_user}:")
for i, rec in enumerate(recs, 1):
    print(f"{i}. {rec['shop_name']} (predicted rating: {rec['predicted_rating']:.2f})")

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('ridge_model_personalized.pkl', 'wb') as f:
    pickle.dump(ridge, f)