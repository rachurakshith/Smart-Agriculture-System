import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("data/processed/final_irrigation_dataset.csv")

# ===============================
# SELECT FEATURES
# ===============================
FEATURES = [
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Cloud_Cover",
    "Pressure",
    "Rain_Binary",
    "Moisture",
    "Soil Type"
]

TARGET = "Irrigation_Needed"

# ===============================
# FIX TARGET COLUMN
# ===============================
df[TARGET] = df[TARGET].replace({
    "Yes": 1, "YES": 1,
    "No": 0,  "NO": 0
}).astype(int)

# ===============================
# 🔥 ENCODE SOIL TYPE (CRITICAL FIX)
# ===============================
soil_encoder = LabelEncoder()
df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])

# ===============================
# DOMAIN LOGIC (SMART AGRI RULES)
# ===============================
df.loc[df["Moisture"] < 15, TARGET] = 1
df.loc[(df["Moisture"] > 45) & (df["Rain_Binary"] == 1), TARGET] = 0

# ===============================
# FINAL DATA
# ===============================
df = df[FEATURES + [TARGET]]

X = df[FEATURES]
y = df[TARGET]

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# MODEL (IMBALANCE HANDLED)
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ===============================
# SAVE EVERYTHING
# ===============================
joblib.dump(model, "models/irrigation_model.pkl")
joblib.dump(FEATURES, "models/feature_order.pkl")
joblib.dump({"Soil Type": soil_encoder}, "models/encoders.pkl")

print("✅ Model retrained successfully")
print("✅ Features:", FEATURES)
print("✅ Soil Type classes:", list(soil_encoder.classes_))
