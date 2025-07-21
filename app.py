
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Coffee Shop Recommender")
st.caption("AIPI 540 Final Project – Evan Moh")

# Load Models and data 
@st.cache_resource
def load_all():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('ridge_model_personalized.pkl', 'rb') as f:
        ridge = pickle.load(f)

    with open('shop_profiles.pkl', 'rb') as f:
        shop_profiles = pickle.load(f)
    data = pd.read_csv('coffeeshop_data_cleaned.csv')
    return tfidf, ridge, shop_profiles, data

tfidf, ridge, shop_profiles, coffee_shops = load_all()
def analyze_review_sentiment(text, rating):
    text = text.lower()
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
    coffee_words = ['coffee', 'espresso', 'latte', 'cappuccino', 'brew', 'roast']
    if any(word in text for word in coffee_words):
        if rating >= 4:
            features['coffee_quality'] = 1
        elif rating <= 2:
            features['coffee_quality'] = -1
    food_words = ['food', 'breakfast', 'lunch', 'sandwich', 'pastry', 'meal']
    if any(word in text for word in food_words):
        if rating >= 4:
            features['food_quality'] = 1
        elif rating <= 2:
            features['food_quality'] = -1
    service_words = ['service', 'staff', 'server', 'waitress', 'friendly', 'rude']
    if any(word in text for word in service_words):
        if rating >= 4:
            features['service_quality'] = 1
        elif rating <= 2:
            features['service_quality'] = -1
    atmosphere_words = ['atmosphere', 'vibe', 'ambiance', 'cozy', 'loud', 'quiet']
    if any(word in text for word in atmosphere_words):
        if rating >= 4:
            features['atmosphere'] = 1
        elif rating <= 2:
            features['atmosphere'] = -1
    work_words = ['wifi', 'work', 'study', 'laptop', 'internet']
    if any(word in text for word in work_words):
        if rating >= 4:
            features['wifi_work'] = 1
        elif rating <= 2:
            features['wifi_work'] = -1
    space_words = ['space', 'seating', 'crowded', 'room', 'tables']
    if any(word in text for word in space_words):
        if rating >= 4:
            features['space_seating'] = 1
        elif rating <= 2:
            features['space_seating'] = -1
    price_words = ['price', 'expensive', 'cheap', 'cost', 'money']
    if any(word in text for word in price_words):
        if rating >= 4:
            features['pricing'] = 1
        elif rating <= 2:
            features['pricing'] = -1
    location_words = ['location', 'convenient', 'parking', 'accessible']
    if any(word in text for word in location_words):
        if rating >= 4:
            features['location'] = 1
        elif rating <= 2:
            features['location'] = -1
    return features


def build_features(user_text, shop_id):
    # User features (from their review)
    user_aspects_dict = analyze_review_sentiment(user_text, 3)  # Assume neutral rating for aspects extraction
    user_aspects = np.array([user_aspects_dict[f] for f in [
        'coffee_quality', 'food_quality', 'service_quality', 
        'atmosphere', 'wifi_work', 'space_seating', 'pricing', 'location'
    ]])
    user_tfidf = tfidf.transform([user_text]).toarray()[0]

    # Shop features (from saved shop_profiles)
    shop_aspects, shop_tfidf = shop_profiles[shop_id]

    features = np.concatenate([user_aspects, user_tfidf, shop_aspects, shop_tfidf])
    return features.reshape(1, -1)


# --- Example Reviews ---
st.subheader("Try an Example Review")
example_reviews = [
    "I loved the cozy atmosphere and the coffee was top-notch. Friendly staff too!",
    "Service was slow and my cappuccino tasted burnt. Not coming back.",
    "Good place for studying with fast wifi and lots of outlets. Food was decent."
]

if 'input_text' not in st.session_state:
    st.session_state.input_text = ''

def fill_example(example):
    st.session_state.input_text = example

cols = st.columns(3)
for idx, example in enumerate(example_reviews):
    with cols[idx]:
        st.button(f"Example {idx+1}", on_click=fill_example, args=(example,))

# --- User Input ---
st.subheader("Write Your Own Review")
user_review = st.text_area("Paste or write a coffee shop review here:", value=st.session_state.input_text, height=120)


#  Recommend Top N Shops 
from sklearn.metrics.pairwise import cosine_similarity

def recommend_shops_cosine(user_text, top_n=5):
    user_aspects_dict = analyze_review_sentiment(user_text, 3)
    user_aspects = np.array([user_aspects_dict[f] for f in [
        'coffee_quality', 'food_quality', 'service_quality', 
        'atmosphere', 'wifi_work', 'space_seating', 'pricing', 'location'
    ]])
    user_tfidf = tfidf.transform([user_text]).toarray()[0]
    user_vector = np.concatenate([user_aspects, user_tfidf])
    
    similarities = []
    for shop_id, (shop_aspects, shop_tfidf) in shop_profiles.items():
        shop_vector = np.concatenate([shop_aspects, shop_tfidf])
        sim = cosine_similarity([user_vector], [shop_vector])[0][0]
        shop_name = coffee_shops[coffee_shops['id'] == shop_id]['name'].iloc[0]
        shop_row = coffee_shops[coffee_shops['id'] == shop_id]
        similarities.append({
            'name': shop_name,
            'similarity': sim,
            'location': shop_row['address'].iloc[0] if 'address' in shop_row else 'N/A',
            'yelp_rating': shop_row['rating'].iloc[0]
        })
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_n]


# Run Recommender
if st.button("Recommend Coffee Shops"):
    if not user_review.strip():
        st.warning("Please enter a review or click an example.")
    else:
        review_length = len(user_review.split())
        recs = recommend_shops_cosine(user_review, top_n=5)
        st.subheader("Top Recommendations for You")
        for i, rec in enumerate(recs, 1):
            st.markdown(f"**{i}. {rec['name']}**  \n"
                        f"Similarity Score: **{rec['similarity']:.2f}**  \n"
                        f"Yelp Rating: {rec['yelp_rating']}  \n"
                        f"Location: {rec['location']}")
        st.info("These suggestions are based on your review and shop features using a classical ML (Ridge regression) model.")
