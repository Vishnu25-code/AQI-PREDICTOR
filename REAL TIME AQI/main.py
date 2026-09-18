import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Ignore warnings
warnings.filterwarnings("ignore")

# Constants
DATA_PATH = r"C:\Users\chauh\OneDrive\Desktop\REAL TIME AQI\aqi_data.csv"
MODEL_PATH = r"C:\Users\chauh\OneDrive\Desktop\REAL TIME AQI\aqi_model.pkl"

def load_and_preprocess_data(path):
    """Loads AQI data and performs cleaning and preprocessing."""
    print("Loading data...")
    df = pd.read_csv(path)

    # Drop rows where AQI is missing
    df = df.dropna(subset=["AQI"])

    # Fill missing numeric columns with mean
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fix extra unnamed columns
    df = df.drop(columns=["Unnamed: 9", "Unnamed: 10"], errors="ignore")

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Create Month and Year columns for analysis
    df["Month"] = df["Date"].dt.month
    df["Year"]  = df["Date"].dt.year

    return df

def perform_eda(df):
    """Performs basic Exploratory Data Analysis."""
    print("Performing EDA...")

    # AQI Distribution
    plt.figure(figsize=(7,4))
    plt.hist(df['AQI'], bins=30, color='skyblue', edgecolor='black')
    plt.title("AQI Distribution")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.show()

    # Average AQI by Month
    plt.figure(figsize=(8,5))
    monthly = df.groupby("Month")["AQI"].mean()
    monthly.plot(kind="bar", color='orange')
    plt.title("Average AQI by Month")
    plt.xlabel("Month")
    plt.ylabel("Average AQI")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(8,5))
    cols = ['PM2.5','PM10','NO2','SO2','CO','O3','AQI']
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def train_aqi_model(df):
    """Trains a RandomForest model to predict AQI."""
    print("Training model...")
    features = ['PM2.5','PM10','NO2','SO2','CO','O3']
    X = df[features]
    y = df['AQI']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Trained.\nRMSE: {rmse:.2f}\nR² Score: {r2:.2f}")

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model

def predict_realtime_aqi(model, new_data):
    """Predicts AQI for new input data."""
    # Ensure new_data is a DataFrame with correct feature names
    if isinstance(new_data, list):
        features = ['PM2.5','PM10','NO2','SO2','CO','O3']
        new_data = pd.DataFrame(new_data, columns=features)

    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    # 1. Load and Preprocess
    df = load_and_preprocess_data(DATA_PATH)

    # 2. EDA (Optional - comment out if not needed)
    # perform_eda(df)

    # 3. Train and Save Model
    model = train_aqi_model(df)

    # 4. Test Real-time Prediction
    print("\n--- Testing Real-time Prediction ---")
    test_inputs = [
        [180, 350, 140, 55, 1.1, 300],
        [200, 400, 160, 60, 1.3, 340],
        [220, 380, 150, 50, 1.2, 320]
    ]

    preds = predict_realtime_aqi(model, test_inputs)

    for i, p in enumerate(preds):
        print(f"Input {i+1}: {test_inputs[i]} -> Predicted AQI: {p:.2f}")
