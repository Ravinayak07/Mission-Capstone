import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and scaler
model = pickle.load(open("Models/model.sav", "rb"))
scaler = pickle.load(open("Models/scaler.sav", "rb"))


# Function to preprocess input data
def preprocess_data(data):
    feature_names = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "isFlaggedFraud",
    ]

    data["type"] = data["type"].map(
        {"CASH_OUT": 5, "PAYMENT": 4, "CASH_IN": 3, "TRANSFER": 2, "DEBIT": 1}
    )

    # Feature scaling
    data_scaled = scaler.transform(data[feature_names])

    return data_scaled


# Streamlit App
def main():
    st.title("Fraud Transaction Detection App")

    # Sidebar title
    st.sidebar.title("User Input")

    # User input for transaction details
    step = st.sidebar.number_input("Step", min_value=1)
    type_val = st.sidebar.selectbox(
        "Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
    )
    amount = st.sidebar.number_input("Amount", value=0.0)
    oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", value=0.0)
    oldbalanceDest = st.sidebar.number_input("Old Balance Destination", value=0.0)
    isFlaggedFraud = st.sidebar.checkbox("Flagged Fraud")

    # Submit button
    if st.sidebar.button("Submit"):
        # Create a DataFrame with user input
        user_data = pd.DataFrame(
            {
                "step": [step],
                "type": [type_val],
                "amount": [amount],
                "oldbalanceOrg": [oldbalanceOrg],
                "oldbalanceDest": [oldbalanceDest],
                "isFlaggedFraud": [isFlaggedFraud],
            }
        )

        # Preprocess the user input
        user_data_scaled = preprocess_data(user_data)

        # Make a prediction
        prediction = model.predict(user_data_scaled)

        # Display the result
        st.header("Prediction:")
        if prediction[0] == 1:
            st.error("This transaction is predicted as Fraud!")
        else:
            st.success("This transaction is predicted as Not Fraud.")


if __name__ == "__main__":
    main()
