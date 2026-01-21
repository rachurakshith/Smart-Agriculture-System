import pandas as pd

# Load datasets
weather = pd.read_csv(r"data/raw weather/weather_data.csv")
soil = pd.read_csv(r"C:\smart_irrigation_system\data\soil_data.csv")

# Add Rain binary
weather["Rain_Binary"] = weather["Rain"].apply(lambda x: 1 if x == "rain" else 0)

# Merge datasets (feature-based merge)
data = pd.concat([weather, soil], axis=1)

# Create target variable
def irrigation_logic(row):
    if row["Moisture"] < 40 and row["Temperature"] > 28 and row["Rain_Binary"] == 0:
        return "Yes"
    else:
        return "No"

data["Irrigation_Needed"] = data.apply(irrigation_logic, axis=1)

# Save final dataset
data.to_csv("data/processed/final_irrigation_dataset.csv", index=False)

print("Final dataset created successfully")
