
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. LOAD TRAINED MODEL AND SCALER
# --------------------------------

model = joblib.load("machine_failure_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

# --------------------------------
# 2. PAGE TITLE
# --------------------------------

st.title("Industrial Machine Failure Prediction")
st.write(
    "Predict possible machine failure using machine operating parameters."
)

st.info(
    "Enter the operating conditions below and click Predict."
)

# --------------------------------
# 3. INPUT PARAMETERS
# --------------------------------

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=298.5
)

process_temp = st.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=308.7
)

speed = st.number_input(
    "Rotational Speed (rpm)",
    min_value=0,
    max_value=5000,
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=200.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=100
)

# --------------------------------
# 4. PREDICTION
# --------------------------------

if st.button("Predict Machine Failure", type="primary"):

    # Encode machine type
    type_value = {"L": 0, "M": 1, "H": 2}[machine_type]

    # Create input using the feature names
    # and exact feature order from the trained model
    feature_values = {
        "Type": type_value,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    expected_features = list(model.feature_names_in_)

    missing_features = [
        feature for feature in expected_features
        if feature not in feature_values
    ]

    if missing_features:
        st.error(
            "The model expects additional or differently named features: "
            + ", ".join(missing_features)
        )
        st.write("Expected feature order:", expected_features)

    else:
        input_data = pd.DataFrame(
            [[feature_values[f] for f in expected_features]],
            columns=expected_features
        )

        # Scale only models trained using scaled data
        model_name = type(model).__name__

        if model_name in [
            "LogisticRegression",
            "KNeighborsClassifier"
        ]:
            model_input = scaler.transform(input_data)
        else:
            model_input = input_data

        # Generate prediction
        prediction = model.predict(model_input)[0]

        # Display result
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Machine Failure Predicted")
        else:
            st.success("✅ No Machine Failure Predicted")

        # Display estimated probability when supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]
            failure_probability = probabilities[
                list(model.classes_).index(1)
            ]

            st.metric(
                "Model-estimated Failure Probability",
                f"{failure_probability:.2%}"
            )

        st.caption(
            "Educational ML demonstration using the AI4I dataset. "
            "This prediction is not a certified industrial safety system."
        )

