
import pandas as pd
import joblib
import math

print("========================================")
print("   SMART CAMPUS CANTEEN")
print("   FOOD DEMAND PREDICTION SYSTEM")
print("========================================")

# Load trained model and feature columns
model = joblib.load("models/random_forest_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Available food items
food_items = [
    "Meals",
    "Dosa",
    "Biryani",
    "Samosa",
    "Sandwich",
    "Idli",
    "Parotta",
    "Tea",
    "Coffee",
    "Vegetable Puff"
]

# Food item measurement units
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

# Available weekdays
weekdays = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday"
]

# Display food items
print("\nAvailable Food Items:")

for i, item in enumerate(food_items, start=1):
    print(f"{i}. {item}")

# Select food item
while True:
    try:
        choice = int(input("\nSelect food item number: "))

        if 1 <= choice <= len(food_items):
            selected_food = food_items[choice - 1]
            break

        print("Please select a valid food item.")

    except ValueError:
        print("Please enter a number.")

# Select weekday
print("\nAvailable Days:")

for i, day in enumerate(weekdays, start=1):
    print(f"{i}. {day}")

while True:
    try:
        day_choice = int(input("\nSelect day number: "))

        if 1 <= day_choice <= len(weekdays):
            selected_day = weekdays[day_choice - 1]
            break

        print("Please select a valid day.")

    except ValueError:
        print("Please enter a number.")

# Function to validate binary inputs
def get_binary_input(prompt):
    while True:
        value = input(prompt).strip()

        if value in ["0", "1"]:
            return int(value)

        print("Invalid input. Please enter only 1 or 0.")

# Get additional information
is_exam_day = get_binary_input(
    "\nIs it an exam day? (1 = Yes, 0 = No): "
)

is_holiday = get_binary_input(
    "Is it a holiday? (1 = Yes, 0 = No): "
)

is_festival = get_binary_input(
    "Is it a festival? (1 = Yes, 0 = No): "
)

# Check holiday
if is_holiday == 1:
    print("\nThe canteen is closed on official holidays.")
    print("No food demand prediction is required.")
    raise SystemExit

# Function to validate non-negative numerical inputs
def get_non_negative_input(prompt):
    while True:
        try:
            value = float(input(prompt))

            if math.isfinite(value) and value >= 0:
                return value

            print("Enter a valid non-negative number.")

        except ValueError:
            print("Please enter a valid number.")

# Validate food price
while True:
    price = get_non_negative_input(
        "\nEnter food item price: "
    )

    if price > 0:
        break

    print("Price must be greater than zero.")

# Validate discount percentage
while True:
    discount = get_non_negative_input(
        "Enter discount percentage (0-100): "
    )

    if discount <= 100:
        break

    print("Discount cannot exceed 100%.")

# Get previous sales information
previous_day_sales = get_non_negative_input(
    "Previous operating day's sales: "
)

previous_week_sales = get_non_negative_input(
    "Sales six operating days ago: "
)

average_sales = get_non_negative_input(
    "Average sales of previous 7 operating days: "
)

# Prepare input data
input_data = pd.DataFrame([{
    "day_of_week": selected_day,
    "is_weekend": 1 if selected_day == "Saturday" else 0,
    "is_exam_day": is_exam_day,
    "is_holiday": is_holiday,
    "is_festival": is_festival,
    "canteen_open": 1,
    "food_item": selected_food,
    "price": price,
    "discount_percentage": discount,
    "previous_day_sales": previous_day_sales,
    "previous_week_sales": previous_week_sales,
    "average_sales_7days": average_sales
}])

# Encode categorical features
input_data = pd.get_dummies(
    input_data,
    columns=["day_of_week", "food_item"],
    dtype=int
)

# Match model training features
input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# Predict demand
predicted_demand = model.predict(input_data)[0]

# Calculate recommended preparation quantity
recommended_quantity = math.ceil(
    predicted_demand * 1.10
)

# Get correct measurement unit
unit = food_units[selected_food]

# Display results
print("\n========================================")
print("          PREDICTION RESULTS")
print("========================================")

print("Food Item:", selected_food)
print("Day:", selected_day)

print(
    "Predicted Demand:",
    round(predicted_demand),
    unit
)

print(
    "Recommended Preparation:",
    recommended_quantity,
    unit
)

print("\nSafety Margin: 10%")
print("========================================")
print("       PREDICTION COMPLETED")
print("========================================")
