import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("best_clv_model.joblib")
scaler = joblib.load("clv_scaler.joblib")

# Set page title
st.set_page_config(page_title="Customer Lifetime Value Prediction")

# Add a title
st.title("Customer Lifetime Value Prediction")

# Create input fields for features
st.header("Enter Customer Information")
transaction_count = st.number_input("Number of Transactions", min_value=1, value=10)
total_quantity = st.number_input(
    "Total Quantity of Items Purchased", min_value=1, value=50
)
recency = st.number_input("Days Since Last Purchase", min_value=0, value=30)
avg_purchase_value = st.number_input(
    "Average Purchase Value", min_value=0.0, value=100.0
)
tenure = st.number_input("Customer Tenure (in days)", min_value=1, value=365)

# Create a dataframe from the input
# Ensure this order matches the order used during model training
input_data = pd.DataFrame(
    {
        "TransactionCount": [transaction_count],
        "TotalQuantity": [total_quantity],
        "Recency": [recency],
        "AvgPurchaseValue": [avg_purchase_value],
        "Tenure": [tenure],
    }
)

# Ensure the column order matches the order used during model training
expected_columns = [
    "TransactionCount",
    "TotalQuantity",
    "Recency",
    "Tenure",
    "AvgPurchaseValue",
]
input_data = input_data[expected_columns]

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make prediction when the user clicks the button
if st.button("Predict Customer Lifetime Value"):
    prediction = model.predict(input_scaled)
    st.success(f"The predicted Customer Lifetime Value is: ${prediction[0]:.2f}")

# Add some information about the model and features
st.sidebar.header("About")
st.sidebar.info(
    "This app predicts the Customer Lifetime Value based on historical transaction data."
)
st.sidebar.header("Features Used")
st.sidebar.markdown(
    """
- Number of Transactions: Total number of purchases made by the customer
- Total Amount Spent: Sum of all purchase amounts
- Total Quantity: Sum of all items purchased
- Recency: Number of days since the last purchase
- Average Purchase Value: Average amount spent per transaction
- Tenure: Number of days since the customer's first purchase
"""
)
