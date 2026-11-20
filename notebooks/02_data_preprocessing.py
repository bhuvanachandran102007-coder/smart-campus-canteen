
import pandas as pd

# Load the dataset
df = pd.read_csv(
    "dataset/canteen_sales_2026_synthetic.csv",
    parse_dates=["date"]
)

print("========== DATA PREPROCESSING ==========")

# Keep only days when the canteen is operating
df = df[df["canteen_open"] == 1].copy()

# Sort records chronologically
df = df.sort_values("date").reset_index(drop=True)

print("\nOperating-day records:", len(df))

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

print("\nEncoded input features:")
print(X.head())

# Select prediction target
y = df["units_sold"]

# Display input features
print("\nInput features (X):")
print(X.head())

# Display target values
print("\nPrediction target (y):")
print(y.head())

# Display shapes
print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)

# Check missing values
print("\nMissing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum())

print("\n========== PREPROCESSING STEP 1 COMPLETED ==========")

# Chronological train-test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== CHRONOLOGICAL TRAIN-TEST SPLIT ==========")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTraining date range:")
print(df["date"].iloc[0], "to", df["date"].iloc[split_index - 1])

print("\nTesting date range:")
print(df["date"].iloc[split_index], "to", df["date"].iloc[-1])

print("\n========== TRAIN-TEST SPLIT COMPLETED ==========")
