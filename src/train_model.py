import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import our custom functions
from data_preprocessing import load_data, preprocess_data
from feature_engineering import create_features

# -------------------------------
# 1. Load and prepare the data
# -------------------------------

# Load dataset
df = load_data("data/youtube_data.csv")

# Clean data
df = preprocess_data(df)

# Create new features
df = create_features(df)

# -------------------------------
# 2. Sort data by time 
# -------------------------------
# This avoids data leakage
# We simulate real-world prediction (past -> future)
df = df.sort_values(by='publish_time')

# -------------------------------
# 3. Select features
# -------------------------------
features = [
    'category_id',
    'publish_hour',
    'publish_day',
    'publish_month',
    'title_length',
    'title_word_count',
    'num_tags'
]

X = df[features]
y = df['log_views']

# -------------------------------
# 4. Time-based split
# -------------------------------
# Use first 80% for training, last 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# -------------------------------
# 5. Train model
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 6. Make predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 7. Evaluate model
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("MSE:", mse)
print("R2:", r2)

# -------------------------------
# 8. Final Insight
# -------------------------------
# If R2 is low or negative:
# → Model cannot predict future views well
# → Features are not strong enough
