import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path):
    """
    Load dataset from a CSV file.
    """
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded dataset: {path}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        exit()

def preprocess_data(df):
    """
    Feature engineering from raw YouTube dataset.
    """

    # -----------------------------
    # 1. Convert publish_time to datetime
    # -----------------------------
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

    # Extract time-based features
    df['publish_hour'] = df['publish_time'].dt.hour
    df['publish_day'] = df['publish_time'].dt.day
    df['publish_month'] = df['publish_time'].dt.month

    # -----------------------------
    # 2. Title features
    # -----------------------------
    df['title_length'] = df['title'].astype(str).apply(len)
    df['title_word_count'] = df['title'].astype(str).apply(lambda x: len(x.split()))

    # -----------------------------
    # 3. Tags feature
    # -----------------------------
    df['num_tags'] = df['tags'].astype(str).apply(lambda x: len(x.split('|')))

    # -----------------------------
    # 4. Select features
    # -----------------------------
    features = [
        'category_id',
        'publish_hour',
        'publish_day',
        'publish_month',
        'title_length',
        'title_word_count',
        'num_tags'
    ]

    target = 'views'

    # -----------------------------
    # 5. Clean data
    # -----------------------------
    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    return X, y

def train_model(X_train, y_train):
    """
    Train Random Forest model.
    """

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("✅ Model training completed")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Performance:")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    return y_pred


def show_feature_importance(model, feature_names):
    """
    Display feature importance scores.
    """

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)

    print("\n🔥 Feature Importance:")
    print(importance)


def find_dataset():
    """
    Automatically find a CSV file inside the data folder.
    """

    if not os.path.exists("data"):
        print("❌ 'data/' folder not found. Run download_data.py first.")
        exit()

    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]

    if not csv_files:
        print("❌ No CSV files found in 'data/' folder.")
        exit()

    # Take the first CSV file found
    data_path = os.path.join("data", csv_files[0])

    return data_path


def main():
    """
    Main pipeline:
    - Find dataset
    - Load data
    - Preprocess
    - Train model
    - Evaluate
    """

    print("🚀 Starting ML pipeline...\n")

    # Step 1: Find dataset automatically
    data_path = find_dataset()

    # Step 2: Load dataset
    df = load_data(data_path)

    # Step 3: Preprocess data
    X, y = preprocess_data(df)

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 5: Train model
    model = train_model(X_train, y_train)

    # Step 6: Evaluate model
    evaluate_model(model, X_test, y_test)

    # Step 7: Feature importance
    show_feature_importance(model, X.columns)


if __name__ == "__main__":
    main()
