
import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("machine_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️"
)

st.title("Industrial Machine Failure Prediction")
st.write("Enter the machine operating parameters.")

# Machine inputs
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature [K]", value=298.5)
process_temp = st.number_input("Process Temperature [K]", value=308.7)
speed = st.number_input("Rotational Speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=100)

st.subheader("Failure Mode Indicators")
st.caption(
    "Set these to 1 only when the corresponding failure mode "
    "is already known. For a normal-input demonstration, leave them at 0."
)

twf = st.selectbox("Tool Wear Failure (TWF)", [0, 1])
hdf = st.selectbox("Heat Dissipation Failure (HDF)", [0, 1])
pwf = st.selectbox("Power Failure (PWF)", [0, 1])
osf = st.selectbox("Overstrain Failure (OSF)", [0, 1])
rnf = st.selectbox("Random Failure (RNF)", [0, 1])

if st.button("Predict Machine Failure"):

    type_value = {"L": 0, "M": 1, "H": 2}[machine_type]

    # Exact feature names and order expected by the model
    input_data = pd.DataFrame([[
        type_value,
        air_temp,
        process_temp,
        speed,
        torque,
        tool_wear,
        twf,
        hdf,
        pwf,
        osf,
        rnf
    ]], columns=[
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ])

    # Apply scaling only for models trained on scaled features
    model_name = model.__class__.__name__

    if model_name in ["LogisticRegression", "KNeighborsClassifier"]:
        model_input = scaler.transform(input_data)
    else:
        model_input = input_data

    prediction = model.predict(model_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Machine Failure Predicted")
    else:
        st.success("No Machine Failure Predicted")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(model_input)[0][1]
        st.metric("Model-estimated failure probability", f"{probability:.2%}")

    st.caption(
        "Educational demonstration only. This model is not a certified "
        "industrial safety or maintenance system."
    )
