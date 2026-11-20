
import pandas as pd
import numpy as np
from datetime import datetime

# Food items, prices, and units
food_items = {
    "Meals": {"price": 80, "unit": "Plates"},
    "Dosa": {"price": 15, "unit": "Pieces"},
    "Biryani": {"price": 100, "unit": "Plates"},
    "Samosa": {"price": 15, "unit": "Pieces"},
    "Sandwich": {"price": 40, "unit": "Pieces"},
    "Idli": {"price": 10, "unit": "Pieces"},
    "Parotta": {"price": 20, "unit": "Pieces"},
    "Tea": {"price": 10, "unit": "Cups"},
    "Coffee": {"price": 15, "unit": "Cups"},
    "Vegetable Puff": {"price": 20, "unit": "Pieces"}
}

# Generate dates for the year 2026
dates = pd.date_range(
    start="2026-01-01",
    end="2026-12-31",
    freq="D"
)

# Official college holidays from the 2026 calendar
holiday_dates = {
    "2026-06-26",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-26",  # Onam and Milad-un-Nabi
    "2026-09-04",  # Krishna Jayanthi
    "2026-09-14",  # Vinayagar Chathurthi
    "2026-10-02",  # Gandhi Jayanthi
    "2026-10-19",  # Ayutha Pooja
    "2026-10-20",  # Vijaya Dasami
    "2026-11-08"   # Deepavali
}

# Confirmed examination dates and periods
exam_dates = (
    set(pd.date_range("2026-07-20", "2026-07-24").strftime("%Y-%m-%d"))
    | {"2026-09-07"}
    | set(pd.date_range("2026-09-21", "2026-09-26").strftime("%Y-%m-%d"))
)

# Festival dates from the college calendar
festival_dates = {
    "2026-06-26",
    "2026-08-26",
    "2026-09-04",
    "2026-09-14",
    "2026-10-19",
    "2026-10-20",
    "2026-11-08"
}

# Generate calendar information
calendar_data = []

for date in dates:
    day_name = date.day_name()
    date_string = date.strftime("%Y-%m-%d")

    is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0
    is_holiday = 1 if date_string in holiday_dates else 0
    is_festival = 1 if date_string in festival_dates else 0
    is_exam_day = 1 if date_string in exam_dates else 0

    # Canteen closes on Sundays and official college holidays
    canteen_open = (
        0 if day_name == "Sunday" or is_holiday == 1 else 1
    )

    calendar_data.append({
        "date": date,
        "day_of_week": day_name,
        "is_weekend": is_weekend,
        "is_exam_day": is_exam_day,
        "is_holiday": is_holiday,
        "is_festival": is_festival,
        "canteen_open": canteen_open
    })

print("Calendar records generated:", len(calendar_data))
print("Official holidays:", sum(day["is_holiday"] for day in calendar_data))
print("Examination days:", sum(day["is_exam_day"] for day in calendar_data))
print("Festival dates:", sum(day["is_festival"] for day in calendar_data))
print(
    "Canteen closed days:",
    sum(day["canteen_open"] == 0 for day in calendar_data)
)

# Base demand for each food item
base_demand = {
    "Meals": 160,
    "Dosa": 90,
    "Biryani": 100,
    "Samosa": 70,
    "Sandwich": 55,
    "Idli": 100,
    "Parotta": 80,
    "Tea": 140,
    "Coffee": 90,
    "Vegetable Puff": 60
}

# Demand variation based on weekdays
weekday_factor = {
    "Monday": 0.90,
    "Tuesday": 1.00,
    "Wednesday": 1.02,
    "Thursday": 1.05,
    "Friday": 1.10,
    "Saturday": 0.70,
    "Sunday": 0.00
}

# Generate synthetic sales data
sales_data = []

for day in calendar_data:
    for item, details in food_items.items():

        if day["canteen_open"] == 0:
            units_sold = 0
            prepared_quantity = 0
            leftover_quantity = 0
            waste_percentage = 0
            discount_percentage = 0

        else:
            demand = base_demand[item] * weekday_factor[day["day_of_week"]]

            # Add random variation to demand
            random_factor = np.random.normal(1.0, 0.08)
            units_sold = max(0, int(round(demand * random_factor)))

            # Generate prepared quantity with a safety margin
            prepared_quantity = int(
                np.ceil(units_sold * np.random.uniform(1.05, 1.15))
            )

            leftover_quantity = prepared_quantity - units_sold

            if prepared_quantity > 0:
                waste_percentage = (
                    leftover_quantity / prepared_quantity
                ) * 100
            else:
                waste_percentage = 0

            discount_percentage = 0

        sales_data.append({
            "date": day["date"],
            "day_of_week": day["day_of_week"],
            "is_weekend": day["is_weekend"],
            "is_exam_day": day["is_exam_day"],
            "is_holiday": day["is_holiday"],
            "is_festival": day["is_festival"],
            "canteen_open": day["canteen_open"],
            "food_item": item,
            "price": details["price"],
            "discount_percentage": discount_percentage,
            "units_sold": units_sold,
            "prepared_quantity": prepared_quantity,
            "leftover_quantity": leftover_quantity,
            "waste_percentage": round(waste_percentage, 2)
        })

print("Food items:", len(food_items))
print("Total calendar days:", len(dates))
print("Total dataset records:", len(sales_data))

# Create DataFrame
df = pd.DataFrame(sales_data)


# Generate historical sales features using previous operating days

df = df.sort_values(["food_item", "date"]).reset_index(drop=True)

df["previous_day_sales"] = 0.0
df["previous_week_sales"] = 0.0
df["average_sales_7days"] = 0.0

for item in food_items:
    item_mask = (
        (df["food_item"] == item) &
        (df["canteen_open"] == 1)
    )

    operating_sales = df.loc[item_mask, "units_sold"]

    df.loc[item_mask, "previous_day_sales"] = (
        operating_sales.shift(1).fillna(0).values
    )

    df.loc[item_mask, "previous_week_sales"] = (
        operating_sales.shift(6).fillna(0).values
    )

    df.loc[item_mask, "average_sales_7days"] = (
        operating_sales.shift(1)
        .rolling(window=7, min_periods=1)
        .mean()
        .fillna(0)
        .values
    )


# Fill unavailable historical values with zero
historical_columns = [
    "previous_day_sales",
    "previous_week_sales",
    "average_sales_7days"
]

df[historical_columns] = df[historical_columns].fillna(0)

# Arrange the dataset columns
column_order = [
    "date",
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
    "average_sales_7days",
    "units_sold",
    "prepared_quantity",
    "leftover_quantity",
    "waste_percentage"
]

df = df[column_order]

# Save dataset
output_path = "dataset/canteen_sales_2026_synthetic.csv"
df.to_csv(output_path, index=False)

print("\nSynthetic sales quantities generated successfully!")
print("Historical sales features generated successfully!")
print("Total columns:", len(df.columns))
print("Dataset saved successfully:", output_path)

# Basic dataset validation
print("\n--- DATASET VALIDATION ---")

print("Missing values:", df.isnull().sum().sum())

print(
    "Duplicate records:",
    df.duplicated(subset=["date", "food_item"]).sum()
)

print(
    "Closed-day sales:",
    df.loc[df["canteen_open"] == 0, "units_sold"].sum()
)

print(
    "Negative sales:",
    (df["units_sold"] < 0).sum()
)

print(
    "Invalid leftover records:",
    (
        (df["leftover_quantity"] < 0)
        | (df["leftover_quantity"] > df["prepared_quantity"])
    ).sum()
)

print(
    "Invalid waste percentages:",
    (
        (df["waste_percentage"] < 0)
        | (df["waste_percentage"] > 100)
    ).sum()
)

print("\nDataset preview:")
print(df.head())

print("\nDataset information:")
print(df.info())
