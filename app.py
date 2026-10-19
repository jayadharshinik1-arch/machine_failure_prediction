
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("machine_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Machine Failure Predictor")

st.title("Industrial Machine Failure Prediction")
st.write("Enter the machine operating parameters below.")

# Input fields
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature [K]", value=298.5)
process_temp = st.number_input("Process Temperature [K]", value=308.7)
speed = st.number_input("Rotational Speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=100)

if st.button("Predict Machine Failure"):

    type_value = {"L": 0, "M": 1, "H": 2}[machine_type]

    # Keep the same feature order used during training
    input_data = pd.DataFrame([[
        type_value,
        air_temp,
        process_temp,
        speed,
        torque,
        tool_wear
    ]], columns=[
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ])

    # Scale input only if the selected model requires it
    if hasattr(model, "predict_proba"):
        if model.__class__.__name__ in [
            "LogisticRegression", "KNeighborsClassifier"
        ]:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
    else:
        prediction = model.predict(input_data)[0]
        probability = None

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Machine Failure Predicted")
    else:
        st.success("No Machine Failure Predicted")

    if probability is not None:
        st.write(f"Model-estimated failure probability: {probability:.2%}")

    st.caption(
        "This is an educational prediction using the trained dataset, "
        "not a certified industrial safety system."
    )
