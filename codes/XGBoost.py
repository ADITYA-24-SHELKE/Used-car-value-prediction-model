import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


# 1. LOAD DATA

data = pd.read_csv("datas/cardekho_dataset.csv")


# 2. FEATURE ENGINEERING

data["brand_model"] = (
    data["brand"].astype(str)
    + "_"
    + data["model"].astype(str)
)


# 3. DATA CLEANING

data = data[data["vehicle_age"] <= 22].copy()

data = data[
    data["brand"].map(data["brand"].value_counts()) >= 40
].copy()


# 4. FEATURES AND TARGET


X = data[
    [
        "vehicle_age",
        "km_driven",
        "max_power",
        "mileage",
        "engine",
        "brand_model",
        "transmission_type"
    ]
]

y = data["selling_price"]


# 5. CATEGORICAL FEATURES


categorical_features = [
    "brand_model",
    "transmission_type"
]



# 6. PREPROCESSOR


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)



# 7. TRAIN-TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# 8. XGBOOST MODEL


xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)



# 9. COMPLETE PIPELINE


model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb_model)
])



# 10. TRAIN


model.fit(X_train, y_train)



# 11. TEST


y_pred = model.predict(X_test)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)




joblib.dump(
    model,
    "car_price_model.pkl"
)

print("\nModel saved successfully!")
print("File: car_price_model.pkl")