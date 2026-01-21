import requests
import joblib
import pandas as pd

# ===============================
# LOAD MODEL, FEATURES, ENCODERS
# ===============================
model = joblib.load("models/irrigation_model.pkl")
FEATURES = joblib.load("models/feature_order.pkl")
encoders = joblib.load("models/encoders.pkl")

print("Model expects features:", FEATURES)

# ===============================
# OPENWEATHER CONFIG
# ===============================
API_KEY = "63de662ad46b4d22caadaa9bc53a4783"   
CITY = "Chennai"

# ===============================
# FETCH LIVE WEATHER
# ===============================
def get_live_weather():
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={CITY}&appid={API_KEY}&units=metric"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    if data.get("cod") != 200:
        raise Exception(f"Weather API error: {data}")

    return {
        "Temperature": data["main"]["temp"],
        "Humidity": data["main"]["humidity"],
        "Wind_Speed": data["wind"]["speed"],
        "Cloud_Cover": data["clouds"]["all"],
        "Pressure": data["main"]["pressure"],
        "Rain_Binary": 1 if "rain" in data else 0
    }

# ===============================
# INPUTS (RAW USER INPUT)
# ===============================
weather = get_live_weather()

soil_type = "Red"      # MUST be string seen during training
moisture = 8           # try 5 / 20 / 60 to test behavior

# ===============================
# ENCODE CATEGORICALS
# ===============================
soil_encoded = encoders["Soil Type"].transform([soil_type])[0]

# ===============================
# BUILD INPUT ROW
# ===============================
row = {
    "Temperature": weather["Temperature"],
    "Humidity": weather["Humidity"],
    "Wind_Speed": weather["Wind_Speed"],
    "Cloud_Cover": weather["Cloud_Cover"],
    "Pressure": weather["Pressure"],
    "Rain_Binary": weather["Rain_Binary"],
    "Moisture": moisture,
    "Soil Type": soil_encoded
}

df = pd.DataFrame([row])
df = df[FEATURES]   # 🔥 CRITICAL: exact order as training

# ===============================
# RULE‑BASED OVERRIDE 🔥
# ===============================
if moisture < 10:
    decision = "YES – Irrigation Needed 💧"
    prob = 1.00
else:
    prob = model.predict_proba(df)[0][1]
    decision = (
        "YES – Irrigation Needed 💧"
        if prob >= 0.35
        else "NO – Irrigation Not Needed 🌱"
    )

# ===============================
# OUTPUT
# ===============================
print("Irrigation Decision:", decision)
print(f"Irrigation Probability: {round(prob * 100, 2)}%")
