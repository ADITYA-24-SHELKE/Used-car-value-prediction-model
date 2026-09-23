import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

data = pd.read_csv("datas/cardekho_dataset.csv")
data["brand_model"] = data["brand"] + "_" + data["model"]
data = data[data["vehicle_age"] <= 22].copy()
data = data[data["brand"].map(data["brand"].value_counts()) >= 40]



X = data[[
    "vehicle_age",  
    "km_driven",
    "max_power",
    "mileage",
    "engine",
    "brand_model",
    "transmission_type",
    
]]

y = data["selling_price"]

numeric_features = [
    "vehicle_age",
    "km_driven",
    "max_power",
    "mileage",
    "engine"

]

categorical_features = [
    "brand_model",
    "transmission_type",
    
    
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Lasso(alpha=1000, max_iter=10000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("MAPE:", mape)
print("Accuracy:", (1 - mape) * 100)