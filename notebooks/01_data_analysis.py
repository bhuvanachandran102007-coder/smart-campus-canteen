
import pandas as pd

# Load the synthetic canteen dataset
df = pd.read_csv(
    "dataset/canteen_sales_2026_synthetic.csv",
    parse_dates=["date"]
)

print("\n========== DATASET OVERVIEW ==========")

# Display dataset dimensions
print("\nDataset shape:", df.shape)

# Display first five records
print("\nFirst five records:")
print(df.head())

# Display last five records
print("\nLast five records:")
print(df.tail())

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display data types
print("\nData types:")
print(df.dtypes)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display statistical summary
print("\nStatistical summary:")
print(df.describe())

# Display available food items
print("\nFood items:")
print(df["food_item"].unique())

print("\n========== ANALYSIS COMPLETED ==========")


print("\n========== FOOD ITEM SALES ANALYSIS ==========")

# Calculate total sales for each food item
food_sales = (
    df.groupby("food_item")["units_sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTotal sales by food item:")
print(food_sales)

# Calculate average sales per calendar day
average_sales = (
    df.groupby("food_item")["units_sold"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage daily sales by food item:")
print(average_sales)

# Identify highest and lowest selling food items
print("\nHighest selling food item:")
print(food_sales.idxmax(), "-", food_sales.max(), "units")

print("\nLowest selling food item:")
print(food_sales.idxmin(), "-", food_sales.min(), "units")

print("\n========== FOOD ITEM ANALYSIS COMPLETED ==========")


import matplotlib.pyplot as plt

# Bar chart of total food-item sales
plt.figure(figsize=(12, 6))

food_sales.plot(kind="bar")

plt.title("Total Sales by Food Item - 2026")
plt.xlabel("Food Item")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


print("\n========== WEEKDAY SALES ANALYSIS ==========")

# Consider only days when the canteen is open
open_days = df[df["canteen_open"] == 1].copy()

# Arrange weekdays in calendar order
weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday"
]

# Calculate average sales for each weekday
weekday_sales = (
    open_days.groupby("day_of_week")["units_sold"]
    .mean()
    .reindex(weekday_order)
)

print("\nAverage sales by weekday:")
print(weekday_sales)

# Create weekday sales chart
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

weekday_sales.plot(kind="bar")

plt.title("Average Food Sales by Weekday - 2026")
plt.xlabel("Day of the Week")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()

print("\n========== WEEKDAY ANALYSIS COMPLETED ==========")


print("\n========== FOOD WASTE ANALYSIS ==========")

# Consider only operating days
open_days = df[df["canteen_open"] == 1].copy()

# Calculate food waste statistics
waste_analysis = (
    open_days.groupby("food_item")
    .agg(
        total_prepared=("prepared_quantity", "sum"),
        total_sold=("units_sold", "sum"),
        total_leftover=("leftover_quantity", "sum")
    )
)

# Calculate overall waste percentage
waste_analysis["waste_percentage"] = (
    waste_analysis["total_leftover"]
    / waste_analysis["total_prepared"] * 100
)

waste_analysis = waste_analysis.sort_values(
    "waste_percentage", ascending=False
)

print("\nFood waste statistics:")
print(waste_analysis.round(2))

# Plot food waste percentage
plt.figure(figsize=(12, 6))

waste_analysis["waste_percentage"].plot(kind="bar")

plt.title("Food Waste Percentage by Food Item - 2026")
plt.xlabel("Food Item")
plt.ylabel("Waste Percentage (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

print("\n========== FOOD WASTE ANALYSIS COMPLETED ==========")

