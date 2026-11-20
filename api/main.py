
from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ==========================================
# LOAD EXISTING ML MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Smart Campus Canteen API",
    description="Food demand prediction API using a Random Forest model trained on synthetic data.",
    version="1.0.0"
)

# ==========================================
# FOOD ITEMS AND UNITS
# ==========================================

food_units = {
    "Meals": "Plates",
    "Dosa": "Pieces",
    "Biryani": "Plates",
    "Samosa": "Pieces",
    "Sandwich": "Pieces",
    "Idli": "Pieces",
    "Parotta": "Pieces",
    "Tea": "Cups",
    "Coffee": "Cups",
    "Vegetable Puff": "Pieces"
}

weekdays = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday"
]

# ==========================================
# INPUT VALIDATION
# ==========================================

class PredictionInput(BaseModel):
    food_item: str
    day_of_week: str
    is_exam_day: bool = False
    is_festival: bool = False
    price: float = Field(gt=0)
    discount_percentage: float = Field(default=0, ge=0, le=100)
    previous_day_sales: float = Field(ge=0)
    previous_week_sales: float = Field(ge=0)
    average_sales_7days: float = Field(ge=0)

# ==========================================
# HOME / HEALTH CHECK
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Smart Campus Canteen API is running",
        "status": "active",
        "documentation": "/docs"
    }

# ==========================================
# FOOD DEMAND PREDICTION
# ==========================================

@app.post("/predict")
def predict_demand(data: PredictionInput):

    if data.food_item not in food_units:
        return {
            "error": "Invalid food item",
            "available_food_items": list(food_units.keys())
        }

    if data.day_of_week not in weekdays:
        return {
            "error": "Invalid day",
            "available_days": weekdays
        }

    # Create input dataframe
    input_data = pd.DataFrame([{
        "day_of_week": data.day_of_week,
        "is_weekend": 0,
        "is_exam_day": int(data.is_exam_day),
        "is_holiday": 0,
        "is_festival": int(data.is_festival),
        "canteen_open": 1,
        "food_item": data.food_item,
        "price": data.price,
        "discount_percentage": data.discount_percentage,
        "previous_day_sales": data.previous_day_sales,
        "previous_week_sales": data.previous_week_sales,
        "average_sales_7days": data.average_sales_7days
    }])

    # Convert categorical features into numerical columns
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

    # Predict demand
    predicted_demand = float(model.predict(input_data)[0])
    predicted_demand = max(0, predicted_demand)

    # Add 10% safety margin
    recommended_quantity = math.ceil(
        predicted_demand * 1.10
    )

    return {
        "project": "Smart Campus Canteen",
        "food_item": data.food_item,
        "day": data.day_of_week,
        "predicted_demand": round(predicted_demand, 2),
        "recommended_preparation": recommended_quantity,
        "unit": food_units[data.food_item],
        "safety_margin": "10%",
        "data_source": "Synthetic dataset"
    }
