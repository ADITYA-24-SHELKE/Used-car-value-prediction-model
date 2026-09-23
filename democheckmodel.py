
import pandas as pd
import numpy as np
import joblib


# ==================================================
# 1. LOAD SAVED MODEL
# ==================================================

model = joblib.load("car_price_model.pkl")

print("Model loaded successfully!\n")


# ==================================================
# 2. REAL-WORLD TEST DATA
# ==================================================
# Only cars for which the listing provides the
# required specifications.
#
# These are 1st-owner listings.
# actual_price = listing asking price
# ==================================================

test_data = pd.DataFrame({

    "vehicle_age": [
        9,   # 2017 Baleno
        8,   # 2018 Swift
        5,   # 2021 Nexon
        3,   # 2023 Nexon
        13,  # 2012 i20
        11,  # 2015 Swift
        2,   # 2024 Creta
        5    # 2021 Creta
    ],

    "km_driven": [
        39000,
        27996,
        50000,
        50000,
        0,       # CHECK/REPLACE with listing KM if needed
        78000,
        20,
        47000
    ],

    "max_power": [
        83,
        85,
        118,
        118,
        83,
        83,
        113,
        113
    ],

    "mileage": [
        21.4,
        18.6,
        17.33,
        17.05,
        18.15,
        20.4,
        17.0,
        17.0
    ],

    "engine": [
        1197,
        1197,
        1199,
        1199,
        1197,
        1197,
        1493,
        1497
    ],

    "brand_model": [
        "Maruti_Baleno",
        "Maruti_Swift",
        "Tata_Nexon",
        "Tata_Nexon",
        "Hyundai_i20",
        "Maruti_Swift",
        "Hyundai_Creta",
        "Hyundai_Creta"
    ],

    "transmission_type": [
        "Automatic",
        "Manual",
        "Automatic",
        "Automatic",
        "Manual",
        "Manual",
        "Manual",
        "Manual"
    ]
})


# ==================================================
# 3. REAL LISTING PRICES
# ==================================================

actual_prices = np.array([
    499000,
    475000,
    775000,
    775000,
    245000,
    295000,
    1190000,
    1095000
])


# ==================================================
# 4. MAKE PREDICTIONS
# ==================================================

predictions = model.predict(test_data)


# ==================================================
# 5. CREATE RESULTS TABLE
# ==================================================

results = test_data.copy()

results["Actual Price"] = actual_prices

results["Predicted Price"] = predictions

results["Absolute Error"] = abs(
    results["Actual Price"]
    - results["Predicted Price"]
)

results["Percentage Error"] = (
    results["Absolute Error"]
    / results["Actual Price"]
) * 100


# ==================================================
# 6. ROUND VALUES
# ==================================================

results["Predicted Price"] = (
    results["Predicted Price"].round(0)
)

results["Absolute Error"] = (
    results["Absolute Error"].round(0)
)

results["Percentage Error"] = (
    results["Percentage Error"].round(2)
)


# ==================================================
# 7. DISPLAY RESULTS
# ==================================================

print("=" * 100)
print("REAL-WORLD MODEL TEST")
print("=" * 100)

print(
    results[
        [
            "brand_model",
            "vehicle_age",
            "km_driven",
            "Actual Price",
            "Predicted Price",
            "Absolute Error",
            "Percentage Error"
        ]
    ].to_string(index=False)
)


# ==================================================
# 8. OVERALL METRICS
# ==================================================

mae = results["Absolute Error"].mean()

mape = results["Percentage Error"].mean()

rmse = np.sqrt(
    np.mean(
        (
            results["Actual Price"]
            - results["Predicted Price"]
        ) ** 2
    )
)


# ==================================================
# 9. PERCENTAGE WITHIN ERROR RANGE
# ==================================================

within_10 = (
    results["Percentage Error"] <= 10
).mean() * 100

within_20 = (
    results["Percentage Error"] <= 20
).mean() * 100

within_30 = (
    results["Percentage Error"] <= 30
).mean() * 100


# ==================================================
# 10. PRINT SUMMARY
# ==================================================

print("\n" + "=" * 100)
print("OVERALL RESULTS")
print("=" * 100)

print(f"MAE:                 ₹{mae:,.2f}")

print(f"RMSE:                ₹{rmse:,.2f}")

print(f"MAPE:                {mape:.2f}%")

print(f"Within ±10%:         {within_10:.2f}%")

print(f"Within ±20%:         {within_20:.2f}%")

print(f"Within ±30%:         {within_30:.2f}%")

print("=" * 100)
