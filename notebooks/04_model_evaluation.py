
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("========== MODEL PERFORMANCE ANALYSIS ==========")

# Load dataset
df = pd.read_csv(
    "dataset/canteen_sales_2026_synthetic.csv",
    parse_dates=["date"]
)

# Keep only operating days
df = df[df["canteen_open"] == 1].copy()

# Sort chronologically
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

y = df["units_sold"]

# Chronological train-test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# Generate predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== EVALUATION RESULTS ==========")

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": np.round(y_pred, 2)
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(comparison.head(15))

# Plot actual vs predicted sales
plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Food Demand")

# Perfect prediction reference line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()
plt.show()

# Calculate prediction errors
errors = y_test.values - y_pred

plt.figure(figsize=(9, 6))

plt.hist(errors, bins=25)

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")

plt.tight_layout()
plt.show()

# Feature importance analysis
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.head(15))

# Plot top 15 important features
plt.figure(figsize=(10, 7))

plt.barh(
    importance.head(15)["Feature"],
    importance.head(15)["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Features Influencing Food Demand")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("\n========== MODEL ANALYSIS COMPLETED ==========")
