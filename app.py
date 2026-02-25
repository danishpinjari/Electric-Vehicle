import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="EV Range Prediction App",
    page_icon="🚗⚡",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("gradient_boosting_model.pkl")

model = load_model()


st.title("🚗⚡ Electric Vehicle Range Prediction")
st.markdown("Predict **Driving Range (km)** using vehicle specifications")


st.sidebar.header("🔧 Vehicle Specifications")

top_speed_kmh = st.sidebar.number_input("Top Speed (km/h)", 80, 300, 160)
battery_capacity_kWh = st.sidebar.number_input("Battery Capacity (kWh)", 20, 150, 55)
torque_nm = st.sidebar.number_input("Torque (Nm)", 100, 1000, 250)
efficiency_wh_per_km = st.sidebar.number_input("Efficiency (Wh/km)", 100, 300, 160)
acceleration_0_100_s = st.sidebar.number_input("0–100 km/h (seconds)", 2.0, 20.0, 8.5)
fast_charging_power_kw_dc = st.sidebar.number_input("Fast Charging Power (kW)", 20, 350, 100)
seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7])
length_mm = st.sidebar.number_input("Length (mm)", 3000, 5500, 4300)
width_mm = st.sidebar.number_input("Width (mm)", 1500, 2500, 1800)
height_mm = st.sidebar.number_input("Height (mm)", 1200, 2200, 1550)


drivetrain = st.sidebar.selectbox("Drivetrain", ["FWD", "RWD", "AWD"])
segment = st.sidebar.selectbox(
    "Segment",
    [
        "A - Mini", "B - Compact", "C - Medium", "D - Large",
        "E - Executive", "F - Luxury", "G - Sports",
        "I - Luxury", "JA - Mini", "JB - Compact",
        "JC - Medium", "JD - Large", "JE - Executive",
        "JF - Luxury", "N - Passenger Van"
    ]
)

car_body_type = st.sidebar.selectbox(
    "Car Body Type",
    [
        "Hatchback", "Sedan", "SUV", "Coupe",
        "Cabriolet", "Liftback Sedan",
        "Station/Estate", "Small Passenger Van"
    ]
)


input_data = {
    'top_speed_kmh': top_speed_kmh,
    'battery_capacity_kWh': battery_capacity_kWh,
    'torque_nm': torque_nm,
    'efficiency_wh_per_km': efficiency_wh_per_km,
    'acceleration_0_100_s': acceleration_0_100_s,
    'fast_charging_power_kw_dc': fast_charging_power_kw_dc,
    'seats': seats,
    'length_mm': length_mm,
    'width_mm': width_mm,
    'height_mm': height_mm,

    # drivetrain
    'drivetrain_AWD': 1 if drivetrain == "AWD" else 0,
    'drivetrain_FWD': 1 if drivetrain == "FWD" else 0,
    'drivetrain_RWD': 1 if drivetrain == "RWD" else 0,
}

# segment one-hot
segments = [
    "A - Mini", "B - Compact", "C - Medium", "D - Large",
    "E - Executive", "F - Luxury", "G - Sports",
    "I - Luxury", "JA - Mini", "JB - Compact",
    "JC - Medium", "JD - Large", "JE - Executive",
    "JF - Luxury", "N - Passenger Van"
]

for seg in segments:
    input_data[f"segment_{seg}"] = 1 if segment == seg else 0

# car body one-hot
body_types = [
    "Cabriolet", "Coupe", "Hatchback", "Liftback Sedan",
    "SUV", "Sedan", "Small Passenger Van", "Station/Estate"
]

for body in body_types:
    input_data[f"car_body_type_{body}"] = 1 if car_body_type == body else 0

input_df = pd.DataFrame([input_data])


st.markdown("## 🔮 Prediction")

if st.button("Predict Range 🚀"):
    prediction = model.predict(input_df)[0]
    st.success(f"✅ **Predicted Driving Range: {prediction:.2f} km**")

    st.progress(min(int(prediction / 600 * 100), 100))


st.markdown("---")
st.markdown("📌 **End-to-End ML Project | EV Range Prediction using Gradient Boosting**")