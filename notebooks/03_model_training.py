
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import os

print("========== RANDOM FOREST MODEL TRAINING ==========")

# Load dataset
df = pd.read_csv(
    "dataset/canteen_sales_2026_synthetic.csv",
    parse_dates=["date"]
)

# Keep only operating days
df = df[df["canteen_open"] == 1].copy()

# Sort data chronologically
df = df.sort_values("date").reset_index(drop=True)

# Select input features
feature_columns = [
    "day_of_week",
    "is_weekend",
    "is_exam_day",
    "is_holiday",
    "is_festival",
    "canteen_open",
    "food_item",
    "price",
    "discount_percentage",
    "previous_day_sales",
    "previous_week_sales",
    "average_sales_7days"
]

X = df[feature_columns]

# Encode categorical features
X = pd.get_dummies(
    X,
    columns=["day_of_week", "food_item"],
    dtype=int
)

# Select target
y = df["units_sold"]

# Chronological train-test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

# Train the model
print("\nTraining the Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

# Save feature column names
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\nTrained model saved successfully!")
print("Feature columns saved successfully!")

print("\n========== MODEL TRAINING COMPLETED ==========")
