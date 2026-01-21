import streamlit as st
import requests
import joblib
import pandas as pd

from src.crop_rules import recommend_crops

# ===============================
# LOAD MODEL + ENCODERS
# ===============================
model = joblib.load("models/irrigation_model.pkl")
encoders = joblib.load("models/encoders.pkl")

FEATURES = list(model.feature_names_in_)

# ===============================
# OPENWEATHER CONFIG
# ===============================
API_KEY = "63de662ad46b4d22caadaa9bc53a4783"   
CITY_WEATHER = "Chennai"

# ===============================
# WEATHER FETCH
# ===============================
def get_live_weather(city):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )
    response = requests.get(url)
    data = response.json()

    if data.get("cod") != 200:
        st.error(f"Weather API Error: {data}")
        st.stop()

    return {
        "Temperature": data["main"]["temp"],
        "Humidity": data["main"]["humidity"],
        "Wind_Speed": data["wind"]["speed"],
        "Cloud_Cover": data["clouds"]["all"],
        "Pressure": data["main"]["pressure"],
        "Rain_Binary": 1 if "rain" in data else 0
    }

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config("Smart Agriculture System", "🌾", layout="wide")
st.title("🌾 Irrigation & Crop Recommendation System")

# ======================================================
# 🔹 SECTION 1: IRRIGATION
# ======================================================
st.header("💧 Irrigation Decision System")

col1, col2 = st.columns(2)

with col1:
    soil_type = st.selectbox(
        "Soil Type",
        encoders["Soil Type"].classes_.tolist()
    )

with col2:
    moisture = st.slider("Soil Moisture (%)", 0, 100, 30)

weather = get_live_weather(CITY_WEATHER)

# ===============================
# WEATHER METRICS
# ===============================
st.subheader("🌦 Live Weather Overview")

w1, w2, w3, w4 = st.columns(4)
w1.metric("🌡 Temperature (°C)", weather["Temperature"])
w2.metric("💧 Humidity (%)", weather["Humidity"])
w3.metric("🌬 Wind Speed", weather["Wind_Speed"])
w4.metric("☁ Cloud Cover", weather["Cloud_Cover"])

# ===============================
# WEATHER BAR CHART
# ===============================
weather_df = pd.DataFrame({
    "Metric": ["Temperature", "Humidity", "Wind Speed", "Cloud Cover"],
    "Value": [
        weather["Temperature"],
        weather["Humidity"],
        weather["Wind_Speed"],
        weather["Cloud_Cover"]
    ]
})
st.bar_chart(weather_df.set_index("Metric"))

# ===============================
# PREDICTION
# ===============================
if st.button("🚀 Predict Irrigation Need"):

    soil_encoded = encoders["Soil Type"].transform([soil_type])[0]

    df = pd.DataFrame([{
        "Temperature": weather["Temperature"],
        "Humidity": weather["Humidity"],
        "Wind_Speed": weather["Wind_Speed"],
        "Cloud_Cover": weather["Cloud_Cover"],
        "Pressure": weather["Pressure"],
        "Rain_Binary": weather["Rain_Binary"],
        "Moisture": moisture,
        "Soil Type": soil_encoded
    }])[FEATURES]

    # ===============================
    # RULE OVERRIDE
    # ===============================
    if moisture < 10:
        st.error("🚨 Soil extremely dry!")
        st.success("✅ Irrigation Needed 💧 (Rule Override)")
        prob = 1.0
    else:
        prob = model.predict_proba(df)[0][1]
        decision = prob >= 0.35

        st.success("✅ Irrigation Needed 💧" if decision else "❌ Irrigation Not Needed 🌱")

    # ===============================
    # PROBABILITY GAUGE
    # ===============================
    st.subheader("🎯 Irrigation Probability")
    st.progress(min(int(prob * 100), 100))
    st.write(f"### **{round(prob * 100, 2)}%**")

    # ===============================
    # SOIL MOISTURE ANALYSIS
    # ===============================
    st.subheader("💧 Soil Moisture Analysis")
    moisture_df = pd.DataFrame({
        "Moisture": [moisture],
        "Ideal Threshold": [30]
    })
    st.line_chart(moisture_df)

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    st.subheader("🧠 Factors Influencing Irrigation")
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

# ======================================================
# 🔹 SECTION 2: CROP RECOMMENDATION
# ======================================================
st.divider()
st.header("🌱 Crop Recommendation Engine")

STATE_CITY_MAP = {
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubli"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala"]
}

c1, c2 = st.columns(2)

with c1:
    state = st.selectbox("Select State", list(STATE_CITY_MAP.keys()))
    soil_for_crop = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clay"])

with c2:
    city = st.selectbox("Select City", STATE_CITY_MAP[state])
    season_ui = st.selectbox("Season", ["Summer 🌞", "Monsoon 🌧️", "Winter ❄️"])

season = season_ui.split()[0].lower()

if st.button("🌾 Recommend Crops"):
    crops = recommend_crops(state, city, season, soil_for_crop)

    if crops:
        st.success("✅ Recommended Crops")
        for crop in crops:
            st.write("🌱", crop)

        # ===============================
        # CROP VISUALIZATION
        # ===============================
        crop_df = pd.DataFrame({
            "Crop": crops,
            "Suitability Score": [100 - i * 10 for i in range(len(crops))]
        })
        st.bar_chart(crop_df.set_index("Crop"))
    else:
        st.warning("No suitable crops found.")
