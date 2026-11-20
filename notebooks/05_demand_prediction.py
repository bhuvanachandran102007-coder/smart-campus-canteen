
import pandas as pd
import joblib
import math

print("========== FOOD DEMAND PREDICTION ==========")

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# Load saved feature columns
feature_columns = joblib.load("models/feature_columns.pkl")

# Enter sample prediction details
input_data = pd.DataFrame([{
    "day_of_week": "Monday",
    "is_weekend": 0,
    "is_exam_day": 0,
    "is_holiday": 0,
    "is_festival": 0,
    "canteen_open": 1,
    "food_item": "Meals",
    "price": 80,
    "discount_percentage": 0,
    "previous_day_sales": 150,
    "previous_week_sales": 155,
    "average_sales_7days": 148
}])

# Encode categorical features
input_data = pd.get_dummies(
    input_data,
    columns=["day_of_week", "food_item"],
    dtype=int
)

# Match training feature columns
input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# Predict food demand
predicted_demand = model.predict(input_data)[0]

# Calculate preparation quantity with 10% safety margin
recommended_quantity = math.ceil(predicted_demand * 1.10)

print("\n========== PREDICTION RESULT ==========")

print("Food Item: Meals")
print("Day: Monday")

print("Predicted Demand:", round(predicted_demand), "plates")

print(
    "Recommended Preparation Quantity:",
    recommended_quantity,
    "plates"
)

print("\n========== PREDICTION COMPLETED ==========")
