import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import re

# Load the coffee shop data
coffee_shops = pd.read_csv('coffeeshop_data_cleaned.csv')

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

combined_features = []
for idx, row in review_features_df.iterrows():
    sentiment_features = [row[name] for name in feature_names]
    tfidf_features = review_tfidf[idx].toarray()[0]
    combined_features.append(sentiment_features + list(tfidf_features))

X_features = np.array(combined_features)
y_ratings = review_features_df['rating'].values

print(f"Combined features shape: {X_features.shape}")

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X_features, y_ratings, test_size=0.2, random_state=42)

# Also split the review data
review_train, review_test = train_test_split(review_features_df, test_size=0.2, random_state=42)

# Function to predict rating using both user preferences and restaurant profiles
def predict_rating_nlp(target_user, target_shop):
    """Predict rating using NLP analysis of user preferences and restaurant qualities"""
    
    # Get target user's preferences
    if target_user not in user_profiles:
        return 3.0
    
    user_pref = user_profiles[target_user]
    
    # Get restaurant's profile from reviews
    if target_shop in restaurant_profiles:
        shop_profile = restaurant_profiles[target_shop]
        
        # Calculate match between user preferences and restaurant qualities
        match_score = 0
        feature_count = 0
        
        for feature in feature_names:
            user_likes = user_pref[feature]
            shop_quality = shop_profile[feature]
            
            if abs(user_likes) > 0.1 and abs(shop_quality) > 0.1:
                # If user likes something and restaurant is good at it, positive match
                # If user dislikes something and restaurant is poor at it, still ok
                match_score += user_likes * shop_quality
                feature_count += 1
        
        if feature_count > 0:
            avg_match = match_score / feature_count
            # Convert match score to rating (3.0 base + adjustment)
            predicted = 3.0 + avg_match
            return max(1.0, min(5.0, predicted))
    
    # Fallback: find similar users
    similarities = []
    candidate_ratings = []
    
    for other_user in user_profiles:
        if other_user != target_user:
            other_pref = user_profiles[other_user]
            
            # Calculate similarity between user preferences
            pref_sim = 0
            for feature in feature_names:
                if abs(user_pref[feature]) > 0.1 and abs(other_pref[feature]) > 0.1:
                    if user_pref[feature] * other_pref[feature] > 0:  # Same sign = similar preference
                        pref_sim += 1
            
            # If similar user has rated target shop
            other_ratings = review_train[review_train['userId'] == other_user]
            shop_rating = other_ratings[other_ratings['shopId'] == target_shop]
            
            if not shop_rating.empty and pref_sim > 0:
                similarities.append(pref_sim)
                candidate_ratings.append(shop_rating['rating'].iloc[0])
    
    if len(candidate_ratings) > 0:
        # Weighted average based on similarity
        weights = np.array(similarities)
        predicted = np.average(candidate_ratings, weights=weights)
        return max(1.0, min(5.0, predicted))
    
    return 3.0

# Test prediction function
predictions = []
actuals = []

print("Testing NLP-based predictions...")
for idx, row in review_test.head(20).iterrows():  
    pred = predict_rating_nlp(row['userId'], row['shopId'])
    predictions.append(pred)
    actuals.append(row['rating'])

rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f'RMSE of NLP predictions: {rmse:.3f}')

# Generate recommendations based on NLP analysis
def recommend_shops_nlp(user_id):
    """Recommend shops based on NLP analysis of user preferences"""
    
    if user_id not in user_profiles:
        return "User not found"
    
    user_pref = user_profiles[user_id]
    user_reviews = reviews[reviews['userId'] == user_id]
    visited_shops = set(user_reviews['shopId'])
    
    recommendations = []
    
    # For each unvisited shop, predict rating
    all_shops = coffee_shops['id'].unique()
    for shop_id in all_shops:
        if shop_id not in visited_shops:
            predicted_rating = predict_rating_nlp(user_id, shop_id)
            shop_name = coffee_shops[coffee_shops['id'] == shop_id]['name'].iloc[0]
            
            recommendations.append({
                'shop_id': shop_id,
                'shop_name': shop_name,
                'predicted_rating': predicted_rating
            })
    
    # Sort by predicted rating
    recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return recommendations[:5]

# Test recommendations
test_user = reviews['userId'].iloc[0]
user_review = reviews[reviews['userId'] == test_user].iloc[0]

print(f"\nTest user: {test_user}")
print(f"Their review: '{user_review['review_text'][:100]}...'")
print(f"Their rating: {user_review['rating']}")
print(f"Their preferences: {user_profiles[test_user]}")

recs = recommend_shops_nlp(test_user)
print("\nNLP-based recommendations:")
if isinstance(recs, list):
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['shop_name']} (predicted: {rec['predicted_rating']:.2f})")

# Show how NLP features work
print(f"\nNLP Feature Analysis:")
sample_review = reviews.iloc[0]
sample_features = analyze_review_sentiment(sample_review['review_text'], sample_review['rating'])
print(f"Review: '{sample_review['review_text'][:80]}...'")
print(f"Rating: {sample_review['rating']}")
print("Extracted preferences:")
for feature, value in sample_features.items():
    if value != 0:
        sentiment = "likes" if value > 0 else "dislikes"
        print(f"  {sentiment} {feature}")