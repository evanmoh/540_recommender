# neuralnet.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load the data
coffee_shops = pd.read_csv('coffeeshop_data_cleaned.csv')
print(f"Loaded {len(coffee_shops)} coffee shops")

# Extract reviews and ratings
reviews_data = []
for i in range(len(coffee_shops)):
    shop_id = coffee_shops.iloc[i]['id']
    shop_name = coffee_shops.iloc[i]['name']
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
print(f"Extracted {len(reviews)} reviews")

# 3. Build TF-IDF features from review text
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
X_tfidf = tfidf.fit_transform(reviews['review_text']).toarray()

# Add more simple features (example: Yelp rating of shop, review length)
shop_ratings = coffee_shops.set_index('id')['rating'].to_dict()
reviews['shop_rating'] = reviews['shopId'].map(shop_ratings)
reviews['review_length'] = reviews['review_text'].apply(lambda x: len(x.split()))

# Combine all features
X_full = np.concatenate([X_tfidf, 
                         reviews[['shop_rating', 'review_length']].values], axis=1)
y = reviews['rating'].values.astype(np.float32)

# Train, test split
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# define neural net model
class neural(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Train the NN
results = []

for hidden_size in [32, 64, 128]:        
    for lr in [0.1, 0.01, 0.001]:       
        class neural(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_size)
                self.fc2 = nn.Linear(hidden_size, 32)
                self.fc3 = nn.Linear(32, 1)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = neural(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n_epochs = 65 

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_t)
            loss = loss_fn(output, y_train_t)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).numpy().flatten()
            rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append((hidden_size, lr, rmse))
        print(f"Hidden: {hidden_size}, LR: {lr}, RMSE: {rmse:.3f}")

# pick the best config
best_hidden, best_lr, best_rmse = min(results, key=lambda x: x[2])
print(f"Best config: Hidden: {best_hidden}, LR: {best_lr}, RMSE: {best_rmse:.3f}")

# retrain model with best settings 
class neural(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, best_hidden)
        self.fc2 = nn.Linear(best_hidden, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = neural(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=best_lr)
loss_fn = nn.MSELoss()
n_epochs = 65  

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = loss_fn(output, y_train_t)
    loss.backward()
    optimizer.step()

# Show some predictions
for i in range(5):
    print(f"Review: {reviews.iloc[i]['review_text'][:80]}...")
    print(f"Actual rating: {y_test[i]:.1f}, Predicted: {preds[i]:.2f}\n")

def recommend_shops_neuralnet(user_id, top_n=5):
    user_reviewed = set(reviews[reviews['userId'] == user_id]['shopId'])
    all_shop_ids = coffee_shops['id'].unique()
    recs = []
    for shop_id in all_shop_ids:
        if shop_id not in user_reviewed:
            # use average TF-IDF vector for this shop (across all users' reviews for this shop)
            shop_review_idxs = reviews[reviews['shopId'] == shop_id].index
            if len(shop_review_idxs) == 0:
                avg_tfidf = np.zeros(tfidf.transform([""]).shape[1])
                avg_review_length = 0
            else:
                avg_tfidf = np.mean(tfidf.transform(reviews.loc[shop_review_idxs, 'review_text']).toarray(), axis=0)
                avg_review_length = reviews.loc[shop_review_idxs, 'review_text'].apply(lambda x: len(x.split())).mean()
            shop_rating = coffee_shops[coffee_shops['id'] == shop_id]['rating'].iloc[0]
            feature_vec = np.concatenate([avg_tfidf, [shop_rating, avg_review_length]])
            x = torch.tensor(feature_vec, dtype=torch.float32)
            with torch.no_grad():

                pred_rating = model(x).item()
            shop_name = coffee_shops[coffee_shops['id'] == shop_id]['name'].iloc[0]
            recs.append({'shop_id': shop_id, 'shop_name': shop_name, 'predicted_rating': pred_rating})
    recs = sorted(recs, key=lambda x: x['predicted_rating'], reverse=True)
    return recs[:top_n]

# Usage:
test_user = reviews['userId'].iloc[0]
recs = recommend_shops_neuralnet(test_user)
print(f"Top recommendations for {test_user}:")
for i, rec in enumerate(recs, 1):
    print(f"{i}. {rec['shop_name']} (predicted: {rec['predicted_rating']:.2f})")

