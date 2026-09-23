import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from lightgbm import LGBMRegressor


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


# Categorical features
categorical_features = [
    "brand_model",
    "transmission_type"
]


# Preprocessing
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


# LightGBM model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=-1
    ))
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
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


# Check training performance
train_pred = model.predict(X_train)

print("\nTrain R²:", r2_score(y_train, train_pred))
print("Test R²:", r2)