
import streamlit as st
import pandas as pd
import joblib
import math

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Campus Canteen",
    page_icon="🍽️",
    layout="wide"
)

# ==========================================
# PREDICTION HISTORY INITIALIZATION
# ==========================================

import os
from datetime import datetime

HISTORY_FILE = "prediction_history.csv"

if "prediction_history" not in st.session_state:

    if os.path.exists(HISTORY_FILE):
        st.session_state.prediction_history = pd.read_csv(
            HISTORY_FILE
        )
    else:
        st.session_state.prediction_history = pd.DataFrame(
            columns=[
                "Date and Time",
                "Food Item",
                "Selected Day",
                "Predicted Demand",
                "Recommended Preparation",
                "Unit"
            ]
        )


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns


model, feature_columns = load_model()

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_dataset():
    return pd.read_csv(
        "dataset/canteen_sales_2026_synthetic.csv",
        parse_dates=["date"]
    )


df = load_dataset()

# ==========================================
# FOOD DETAILS
# ==========================================

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
# APPLICATION TITLE
# ==========================================

st.title("🍽️ Smart Campus Canteen")

st.subheader(
    "Food Demand Prediction and Waste Reduction System"
)

st.write(
    "Predict daily food demand and calculate "
    "the recommended preparation quantity using Machine Learning."
)

# ==========================================
# DEMAND PREDICTION INPUTS
# ==========================================

st.divider()

st.header("📋 Enter Prediction Details")

st.subheader("Food and Calendar Details")

col1, col2 = st.columns(2)

with col1:
    selected_prediction_food = st.selectbox(
        "Select Food Item",
        food_items
    )

    selected_day = st.selectbox(
        "Select Day",
        weekdays
    )

    exam_day = st.selectbox(
        "Is it an Exam Day?",
        ["No", "Yes"]
    )

with col2:
    holiday = st.selectbox(
        "Is it a Holiday?",
        ["No", "Yes"]
    )

    festival = st.selectbox(
        "Is it a Festival?",
        ["No", "Yes"]
    )

st.subheader("Sales and Price Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input(
        "Food Price (₹)",
        min_value=0.01,
        value=80.0
    )

    discount = st.number_input(
        "Discount Percentage",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    previous_day_sales = st.number_input(
        "Previous Operating Day Sales",
        min_value=0,
        value=150
    )

with col2:
    previous_week_sales = st.number_input(
        "Sales Six Operating Days Ago",
        min_value=0,
        value=155
    )

    average_sales = st.number_input(
        "Average Sales of Previous 7 Operating Days",
        min_value=0.0,
        value=148.0
    )

# ==========================================
# PREDICTION
# ==========================================

if st.button("🔮 Predict Food Demand", type="primary"):

    if holiday == "Yes":
        st.warning(
            "The canteen is closed on holidays. "
            "No demand prediction is generated."
        )

    else:
        input_data = pd.DataFrame([{
            "day_of_week": selected_day,
            "is_weekend": 0,
            "is_exam_day": 1 if exam_day == "Yes" else 0,
            "is_holiday": 0,
            "is_festival": 1 if festival == "Yes" else 0,
            "canteen_open": 1,
            "food_item": selected_prediction_food,
            "price": price,
            "discount_percentage": discount,
            "previous_day_sales": previous_day_sales,
            "previous_week_sales": previous_week_sales,
            "average_sales_7days": average_sales
        }])

        encoded_input = pd.get_dummies(
            input_data,
            columns=["day_of_week", "food_item"],
            dtype=int
        )

        encoded_input = encoded_input.reindex(
            columns=feature_columns,
            fill_value=0
        )

        predicted_demand = max(
            0,
            float(model.predict(encoded_input)[0])
        )

        recommended_quantity = math.ceil(
            predicted_demand * 1.10
        )

        unit = food_units[selected_prediction_food]

        # Save prediction to history
        new_prediction = pd.DataFrame([{
            "Date and Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Food Item": selected_prediction_food,
            "Selected Day": selected_day,
            "Predicted Demand": round(predicted_demand),
            "Recommended Preparation": recommended_quantity,
            "Unit": unit
        }])

        st.session_state.prediction_history = pd.concat(
            [
                st.session_state.prediction_history,
                new_prediction
            ],
            ignore_index=True
        )

        st.session_state.prediction_history.to_csv(
            HISTORY_FILE,
            index=False
        )


        st.divider()

        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Food Demand",
            f"{predicted_demand:.0f} {unit}"
        )

        col2.metric(
            "Recommended Preparation",
            f"{recommended_quantity} {unit}"
        )

        st.info(
            "Recommended preparation includes a 10% safety margin."
        )

        if discount > 0:
            st.warning(
                "The training dataset contains only 0% discounts. "
                "Predictions with discounts are not validated."
            )

        # Download Prediction Report
        prediction_report = pd.DataFrame([{
            "Food Item": selected_prediction_food,
            "Selected Day": selected_day,
            "Predicted Demand": round(predicted_demand),
            "Recommended Preparation": recommended_quantity,
            "Unit": unit,
            "Safety Margin Percentage": 10
        }])

        prediction_csv = prediction_report.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=prediction_csv,
            file_name="canteen_prediction_report.csv",
            mime="text/csv"
        )


# ==========================================
# DASHBOARD DATA
# ==========================================

open_df = df[df["canteen_open"] == 1].copy()


# ==========================================
# DASHBOARD FILTERS
# ==========================================

st.divider()

st.header("🔎 Dashboard Filters")

# Food filter
selected_food = st.selectbox(
    "Select Food Item for Dashboard",
    ["All Food Items"] + food_items
)

# Date range filter
st.subheader("📅 Select Date Range")

min_date = open_df["date"].min().date()
max_date = open_df["date"].max().date()

selected_dates = st.date_input(
    "Choose Start and End Dates",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply food filter
if selected_food == "All Food Items":
    filtered_df = open_df.copy()
else:
    filtered_df = open_df[
        open_df["food_item"] == selected_food
    ].copy()

# Apply date filter
if len(selected_dates) == 2:

    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date) &
        (filtered_df["date"].dt.date <= end_date)
    ].copy()

st.info(
    f"Showing data from {len(filtered_df)} records."
)


# ==========================================
# CANTEEN PERFORMANCE OVERVIEW
# ==========================================

st.divider()

st.header("📊 Canteen Performance Overview")

total_sold = int(filtered_df["units_sold"].sum())

total_prepared = int(filtered_df["prepared_quantity"].sum())

total_leftover = int(filtered_df["leftover_quantity"].sum())

if total_prepared > 0:
    overall_waste = (
        total_leftover / total_prepared
    ) * 100
else:
    overall_waste = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Food Sold", f"{total_sold:,}")
col2.metric("Total Food Prepared", f"{total_prepared:,}")
col3.metric("Total Leftover Food", f"{total_leftover:,}")
col4.metric("Overall Food Waste", f"{overall_waste:.2f}%")

# ==========================================
# FOOD-WISE SALES ANALYSIS
# ==========================================

st.divider()

st.header("🍽️ Food-wise Sales Analysis")

food_sales = (
    filtered_df.groupby("food_item")["units_sold"]
    .sum()
    .sort_values(ascending=False)
)

st.subheader("Total Sales by Food Item")

st.bar_chart(food_sales)

st.subheader("Food Sales Ranking")

food_ranking = food_sales.reset_index()

food_ranking.columns = [
    "Food Item",
    "Total Units Sold"
]

food_ranking.index = range(1, len(food_ranking) + 1)

st.dataframe(
    food_ranking,
    use_container_width=True
)

if not food_sales.empty:
    highest_food = food_sales.idxmax()
    lowest_food = food_sales.idxmin()

    col1, col2 = st.columns(2)

    col1.success(f"Highest Selling Item: {highest_food}")
    col2.info(f"Lowest Selling Item: {lowest_food}")

# ==========================================
# FOOD WASTE ANALYSIS
# ==========================================

st.divider()

st.header("♻️ Food Waste Analysis")

waste_analysis = (
    filtered_df.groupby("food_item")
    .agg(
        Total_Prepared=("prepared_quantity", "sum"),
        Total_Leftover=("leftover_quantity", "sum")
    )
)

if not waste_analysis.empty:

    waste_analysis["Waste_Percentage"] = (
        waste_analysis["Total_Leftover"]
        / waste_analysis["Total_Prepared"]
    ) * 100

    waste_analysis = waste_analysis.sort_values(
        "Waste_Percentage",
        ascending=False
    )

    st.subheader("Food Waste Percentage by Item")

    st.bar_chart(
        waste_analysis["Waste_Percentage"]
    )

    st.subheader("Food Waste Details")

    st.dataframe(
        waste_analysis,
        use_container_width=True
    )

    highest_waste_food = waste_analysis[
        "Waste_Percentage"
    ].idxmax()

    highest_waste_percentage = waste_analysis.loc[
        highest_waste_food,
        "Waste_Percentage"
    ]

    st.warning(
        f"Highest Waste Item: {highest_waste_food} "
        f"({highest_waste_percentage:.2f}%)"
    )

# ==========================================
# DAILY SALES TREND
# ==========================================

st.divider()

st.header("📈 Daily Sales Trend")

daily_sales = (
    filtered_df.groupby("date")["units_sold"]
    .sum()
    .reset_index()
)

daily_sales.columns = [
    "Date",
    "Total Units Sold"
]

st.subheader("Daily Food Sales")

st.line_chart(
    daily_sales.set_index("Date")["Total Units Sold"]
)

st.subheader("Daily Sales Records")

st.dataframe(
    daily_sales,
    use_container_width=True
)

if not daily_sales.empty:

    highest_sales_day = daily_sales.loc[
        daily_sales["Total Units Sold"].idxmax()
    ]

    st.success(
        f"Highest Sales Day: "
        f"{highest_sales_day['Date'].strftime('%d-%m-%Y')} "
        f"({int(highest_sales_day['Total Units Sold']):,} units)"
    )


# ==========================================
# WEEKDAY SALES INSIGHTS
# ==========================================

st.divider()

st.header("📅 Weekday Sales Insights")

# Calculate total sales for each operating day
daily_weekday_sales = (
    filtered_df.groupby(["date", "day_of_week"])["units_sold"]
    .sum()
    .reset_index()
)

# Calculate average sales for each weekday
weekday_average = (
    daily_weekday_sales.groupby("day_of_week")["units_sold"]
    .mean()
    .reindex(weekdays)
    .dropna()
)

st.subheader("Average Sales by Weekday")

st.bar_chart(weekday_average)

# Identify the busiest weekday
if not weekday_average.empty:

    busiest_day = weekday_average.idxmax()
    busiest_average = weekday_average.max()

    st.success(
        f"Busiest Day: {busiest_day} "
        f"(Average {busiest_average:.0f} units)"
    )

# Display weekday sales details
weekday_details = weekday_average.reset_index()

weekday_details.columns = [
    "Weekday",
    "Average Units Sold"
]

st.dataframe(
    weekday_details,
    use_container_width=True
)

# ==========================================
# MONTHLY SALES INSIGHTS
# ==========================================

st.divider()

st.header("📆 Monthly Sales Insights")

# Calculate total sales for each month
monthly_sales = (
    filtered_df
    .groupby(pd.Grouper(key="date", freq="MS"))["units_sold"]
    .sum()
)

# Display monthly sales chart
st.subheader("Total Food Sales by Month")

st.bar_chart(monthly_sales)

# Identify highest and lowest sales months
if not monthly_sales.empty:

    highest_month = monthly_sales.idxmax()
    lowest_month = monthly_sales.idxmin()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Highest Sales Month",
            highest_month.strftime("%B %Y"),
            f"{int(monthly_sales.max())} units"
        )

    with col2:
        st.metric(
            "Lowest Sales Month",
            lowest_month.strftime("%B %Y"),
            f"{int(monthly_sales.min())} units"
        )

# Display monthly sales details
monthly_details = monthly_sales.reset_index()

monthly_details.columns = [
    "Month",
    "Total Units Sold"
]

monthly_details["Month"] = (
    monthly_details["Month"].dt.strftime("%B %Y")
)

st.dataframe(
    monthly_details,
    use_container_width=True
)

# ==========================================
# EXAM-DAY SALES INSIGHTS
# ==========================================

st.divider()

st.header("📝 Exam-Day vs Normal-Day Sales")

# Calculate total sales for each operating date
exam_daily_sales = (
    filtered_df.groupby(["date", "is_exam_day"])["units_sold"]
    .sum()
    .reset_index()
)

# Calculate average daily sales
exam_comparison = (
    exam_daily_sales.groupby("is_exam_day")["units_sold"]
    .mean()
)

normal_sales = exam_comparison.get(0, 0)
exam_sales = exam_comparison.get(1, 0)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Normal-Day Average Sales",
        f"{normal_sales:.0f} units"
    )

with col2:
    st.metric(
        "Exam-Day Average Sales",
        f"{exam_sales:.0f} units"
    )

# Compare sales between the two day types
if normal_sales > 0:

    difference = exam_sales - normal_sales
    percentage_change = (difference / normal_sales) * 100

    if difference > 0:
        st.success(
            f"Exam-day sales are {percentage_change:.1f}% "
            "higher than normal-day sales."
        )

    elif difference < 0:
        st.info(
            f"Exam-day sales are {abs(percentage_change):.1f}% "
            "lower than normal-day sales."
        )

    else:
        st.info("Sales are equal on both day types.")

else:
    st.warning("Not enough normal-day sales data for comparison.")

# Display comparison chart
comparison_chart = pd.Series({
    "Normal Days": normal_sales,
    "Exam Days": exam_sales
})

st.subheader("Average Daily Sales Comparison")
st.bar_chart(comparison_chart)

# ==========================================
# FOOD DEMAND RANKING
# ==========================================

st.divider()

st.header("🏆 Food Demand Ranking")

# Calculate food-wise demand
food_demand_ranking = (
    open_df.groupby("food_item")
    .agg(
        Total_Sales=("units_sold", "sum"),
        Average_Daily_Sales=("units_sold", "mean"),
        Total_Days=("date", "nunique")
    )
    .sort_values("Total_Sales", ascending=False)
    .reset_index()
)

# Add ranking
food_demand_ranking.insert(
    0,
    "Rank",
    range(1, len(food_demand_ranking) + 1)
)

# Display ranking
st.subheader("Overall Food Demand Ranking")

st.dataframe(
    food_demand_ranking,
    use_container_width=True,
    hide_index=True
)

# Display highest and lowest demand foods
if not food_demand_ranking.empty:

    highest_food = food_demand_ranking.iloc[0]
    lowest_food = food_demand_ranking.iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Most Demanded Food",
            highest_food["food_item"],
            f"{int(highest_food['Total_Sales'])} units sold"
        )

    with col2:
        st.metric(
            "Least Demanded Food",
            lowest_food["food_item"],
            f"{int(lowest_food['Total_Sales'])} units sold"
        )

# ==========================================
# DOWNLOADABLE SALES REPORT
# ==========================================

st.divider()

st.header("📥 Download Sales Report")

st.write(
    "Download the sales data currently selected "
    "in the dashboard."
)

# Prepare report data
report_df = filtered_df.copy()

# Convert date into readable format
report_df["date"] = report_df["date"].dt.strftime("%Y-%m-%d")

# Convert report into CSV
csv_data = report_df.to_csv(index=False).encode("utf-8")

# Download button
st.download_button(
    label="Download Sales Report (CSV)",
    data=csv_data,
    file_name="campus_canteen_sales_report.csv",
    mime="text/csv"
)


# ==========================================
# PREDICTION HISTORY
# ==========================================

st.divider()

st.header("📜 Prediction History")

if not st.session_state.prediction_history.empty:

    st.write(
        "Previously generated food-demand predictions."
    )

    st.dataframe(
        st.session_state.prediction_history,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="📥 Download Prediction History",
        data=st.session_state.prediction_history.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="canteen_prediction_history.csv",
        mime="text/csv"
    )

    # Clear prediction history
    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = pd.DataFrame(
            columns=[
                "Date and Time",
                "Food Item",
                "Selected Day",
                "Predicted Demand",
                "Recommended Preparation",
                "Unit"
            ]
        )

        st.session_state.prediction_history.to_csv(
            HISTORY_FILE,
            index=False
        )

        st.success("Prediction history cleared successfully!")

        st.rerun()

else:
    st.info(
        "No prediction history available yet. "
        "Generate your first prediction."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart Campus Canteen | Food Demand Prediction "
    "and Waste Reduction System"
)

st.caption(
    "Academic ML Project | Predictions and dashboard "
    "statistics are based on a synthetic 2026 dataset, "
    "not actual college sales."
)
