# Washington DC Coffee Shop Recommender

AIPI 540 – Deep Learning | Duke University  
**Author:** Evan Moh

---

## Overview

This project recommends coffee shops in Washington, DC by analyzing user Yelp reviews, ratings, and shop attributes. It combines content-based filtering, machine learning, and neural networks to match users with coffee shops that best fit their inferred preferences. The target variable is the actual user review rating.

---

## Dataset

- **Source:** Yelp (20 hand-picked DC coffee shops)
- **Reviews:** ~5 positive & ~5 negative reviews per shop (manually collected)
- **Features:** Dog-friendly, wifi, parking, vegan options, outdoor seating, etc.
- **Preprocessing:**
  - Fill missing/“no” values with 0 for consistency
  - Extract additional features from profile text
  - Generate aspect scores (e.g., coffee quality, food, service, price) via sentiment and keywords

---

## Modeling Approach

- **Naive Baseline:** Recommends based on overall Yelp rating only.
- **ML Approach (Ridge Regression):**  
  - Uses aspect scores and TF-IDF features  
  - Trains Ridge regression to predict review ratings  
  - Recommends shops with highest predicted ratings for each user
- **Neural Network:**  
  - Feedforward neural network (PyTorch)
  - Inputs: aspect scores + TF-IDF  
  - Three hidden layers (64 → 32 → 1, with ReLU activation)
  - Tuned with early stopping and optimized hyperparameters
- **Data Prep:**  
  - Data cleaning and feature engineering for ML and neural net models

---

## Evaluation

- **Metric:** RMSE (Root Mean Squared Error) between predicted and true ratings
- **Comparison:**  
  - Naive: Shop average only  
  - Ridge Regression: Aspect + TF-IDF  
  - Neural Net: Same features, more flexible model  
- **Result:** Ridge regression achieved the lowest RMSE, likely due to the small dataset

---

## Ethics Statement

- This project is for **educational and research purposes only**.
- All data is anonymized; no personally identifiable information is used.
- The recommender is designed to respect user privacy and avoid bias or discrimination.
- Not intended for real-world deployment or commercial use.

---

## File Structure

<pre> ## File Structure ``` . ├── app.py # Streamlit web application ├── data_prep.py # Data cleaning and feature extraction steps ├── ml.py # Classical ML model (Ridge regression) ├── naive.py # Baseline recommender (Yelp average rating) ├── neuralnet.py # Feedforward neural network recommender (PyTorch) ├── LICENSE ├── README.md ├── requirements.txt ├── data/ │ ├── coffeeshop_data_cleaned.csv │ └── coffeeshop_data.xlsx ├── pickles/ │ ├── ridge_model_personalized.pkl │ ├── ridge_model.pkl │ ├── shop_profiles.pkl │ └── tfidf_vectorizer.pkl ``` </pre>

## File Descriptions

- `app.py` — Main Streamlit application for user interaction and recommendations.
- `data_prep.py` — Script for data cleaning, preprocessing, and feature extraction.
- `ml.py` — Machine learning pipeline using Ridge regression for prediction and recommendation.
- `naive.py` — Baseline approach using only shop average Yelp ratings.
- `neuralnet.py` — Feedforward neural network implementation (PyTorch) for content-based recommendations.
- `data/` — Raw and cleaned data files.
- `pickles/` — Serialized model and feature files (pickle format) used for predictions.

---

## How to Run

1. **Clone this repository:**
   git clone https://github.com/yourusername/coffee-shop-recommender.git
   cd coffee-shop-recommender

2. **Install dependencies:**    
    pip install -r requirements.txt

3. **Check required data/model files:**  
   Ensure the following files are in the correct folders:
   - `data/coffeeshop_data_cleaned.csv`
   - `pickles/tfidf_vectorizer.pkl`
   - `pickles/ridge_model_personalized.pkl`
   - `pickles/shop_profiles.pkl`

4. **Run the app:**
   ```bash
   streamlit run app.py