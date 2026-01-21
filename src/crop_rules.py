# src/crop_rules.py

# ===============================
# LOCATION + SEASON RULES
# ===============================
CROP_RULES = {
    "tamil nadu": {
        "chennai": {
            "summer": ["Groundnut", "Millets", "Sunflower"],
            "monsoon": ["Rice", "Sugarcane", "Maize"],
            "winter": ["Pulses", "Vegetables"]
        },
        "default": {
            "summer": ["Millets", "Groundnut"],
            "monsoon": ["Rice", "Maize"],
            "winter": ["Vegetables"]
        }
    },
    "karnataka": {
        "default": {
            "summer": ["Ragi", "Groundnut"],
            "monsoon": ["Maize", "Paddy"],
            "winter": ["Wheat", "Pulses"]
        }
    },
    "punjab": {
        "default": {
            "summer": ["Cotton"],
            "monsoon": ["Rice"],
            "winter": ["Wheat", "Mustard"]
        }
    }
}

# ===============================
# SOIL → CROP FILTER
# ===============================
SOIL_CROP_MAP = {
    "Sandy": ["Groundnut", "Millets"],
    "Loamy": ["Rice", "Wheat", "Maize", "Sugarcane"],
    "Clay": ["Rice", "Jute"]
}

# ===============================
# RECOMMENDATION ENGINE
# ===============================
def recommend_crops(state, city, season, soil_type):
    state = state.lower()
    city = city.lower()

    state_data = CROP_RULES.get(state, {})
    city_data = state_data.get(city, state_data.get("default", {}))
    season_crops = city_data.get(season, [])

    soil_crops = SOIL_CROP_MAP.get(soil_type, [])

    # Final filtered crops
    final = list(set(season_crops) & set(soil_crops))

    return final if final else season_crops
