import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


# Load dataset
data = pd.read_csv("datas/cardekho_dataset.csv")


# Create combined brand + model feature
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


# Numerical features
numeric_features = [
    "vehicle_age",
    "km_driven",
    "max_power",
    "mileage",
    "engine"
]


# Categorical features
categorical_features = [
    "brand_model",
    "transmission_type"
]


# One-Hot Encoding
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


# Gradient Boosting Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)


print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print("MAPE-based Accuracy:", (1 - mape) * 100)