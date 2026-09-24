from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Bike Rental Demand")

MODEL_PATH = Path(__file__).parent / "bike_model.pkl"
COLS = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "hour", "dow", "month", "year"]
SEASONS = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHERS = {
    1: "Clear / few clouds",
    2: "Mist / cloudy",
    3: "Light snow / light rain",
    4: "Heavy rain / snow / fog",
}


@st.cache_resource
def load_model():
    return pickle.load(open(MODEL_PATH, "rb"))


model = load_model()

st.title("Bike Rental Demand")
st.write(
    "A Random Forest model (trained on the Kaggle Bike Sharing Demand dataset, 2011-2012 Washington D.C.) "
    "predicts how many bikes will be rented in a given hour, from the date, time, and weather."
)

col1, col2 = st.columns(2)
with col1:
    d = st.date_input("Date", date(2012, 6, 15))
    t = st.time_input("Hour", time(17, 0))
    season = st.selectbox("Season", list(SEASONS.keys()), format_func=lambda s: SEASONS[s], index=1)
    weather = st.selectbox("Weather", list(WEATHERS.keys()), format_func=lambda w: WEATHERS[w])
with col2:
    holiday = st.checkbox("Holiday")
    workingday = st.checkbox("Working day", value=True)
    temp = st.slider("Temperature (°C)", -10.0, 45.0, 24.0, 0.5)
    atemp = st.slider("Feels-like temperature (°C)", -10.0, 50.0, 27.0, 0.5)
    humidity = st.slider("Humidity (%)", 0, 100, 55)
    windspeed = st.slider("Wind speed (km/h)", 0.0, 60.0, 12.0, 0.5)

if st.button("Predict demand"):
    row = pd.DataFrame([{
        "season": season,
        "holiday": int(holiday),
        "workingday": int(workingday),
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "hour": t.hour,
        "dow": d.weekday(),
        "month": d.month,
        "year": d.year,
    }])[COLS]

    pred_log = model.predict(row)[0]
    pred = max(0, round(float(np.expm1(pred_log))))
    st.success(f"Predicted rentals in this hour: **{pred}** bikes")

    hours = pd.DataFrame([{**row.iloc[0].to_dict(), "hour": h} for h in range(24)])[COLS]
    hours["predicted_count"] = np.clip(np.expm1(model.predict(hours)), 0, None)
    st.line_chart(hours.set_index(hours.index.astype(str) + ":00")["predicted_count"].rename("Predicted rentals by hour"))

st.caption(
    "Model: Random Forest on log-transformed rental counts (validation RMSLE ≈ 0.31 on a held-out split). "
    "Trained on 2011-2012 data only, so it does not know about longer-term growth or unusual events."
)
