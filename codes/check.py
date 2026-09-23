import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

from xgboost import XGBRegressor


# Load dataset
data = pd.read_csv("datas/cardekho_dataset.csv")


# Create brand + model feature
data["brand_model"] = data["brand"] + "_" + data["model"]


# Data cleaning
data = data[data["vehicle_age"] <= 22].copy()

data = data[
    data["brand"].map(data["brand"].value_counts()) >= 40
].copy()


# Features
X = data[[
    "vehicle_age",
    "km_driven",
    "max_power",
    "mileage",
    "engine",
    "brand_model",
    "transmission_type"
]]


# Target
y = data["selling_price"]


# Categorical features
categorical_features = [
    "brand_model",
    "transmission_type"
]


# Preprocessor
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


# XGBoost model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    ))
])


# 5-Fold Cross Validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# R² scores
r2_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)


# MAE scores
mae_scores = -cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error"
)


# RMSE scores
rmse_scores = np.sqrt(
    -cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_squared_error"
    )
)


# Print individual fold results
print("R² Scores:", r2_scores)
print("MAE Scores:", mae_scores)
print("RMSE Scores:", rmse_scores)


# Print averages
print("\nAverage R²:", r2_scores.mean())
print("Average MAE:", mae_scores.mean())
print("Average RMSE:", rmse_scores.mean())


# Print standard deviation
print("\nR² Standard Deviation:", r2_scores.std())
print("MAE Standard Deviation:", mae_scores.std())
print("RMSE Standard Deviation:", rmse_scores.std())