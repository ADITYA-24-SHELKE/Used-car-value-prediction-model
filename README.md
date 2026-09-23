# 🚗 Used Car Value Prediction Model

A machine learning project that predicts the selling price of used cars using historical CarDekho data and vehicle specifications.

## 🌐 Live Demo

https://used-car-value-prediction-model-yekrg79ihadqvncxpg9pvm.streamlit.app/

## 📌 Project Overview

This project covers the complete machine learning workflow:

- Data exploration and cleaning
- Feature engineering
- Model comparison
- Model evaluation
- XGBoost model training
- Model saving using Joblib
- Tkinter desktop application
- Streamlit web application
- Git & GitHub version control
- Deployment using Streamlit Community Cloud

## 🔍 Features Used

The final model uses:

- Vehicle age
- Kilometers driven
- Maximum power
- Mileage
- Engine capacity
- Brand + model
- Transmission type

## 🤖 Machine Learning Model

The final model uses **XGBoost Regressor** with:

- `n_estimators = 500`
- `learning_rate = 0.03`
- `max_depth = 6`
- `subsample = 0.8`
- `colsample_bytree = 0.8`

### 5-Fold Cross-Validation Results

- **Average R²:** 0.947
- **Average MAE:** ₹90,161
- **Average RMSE:** ₹173,891

## 🛠️ Technologies Used

Python • Pandas • NumPy • Scikit-learn • XGBoost • LightGBM • Joblib • Tkinter • Streamlit • Git • GitHub

## 📂 Project Structure

```text
Used-car-value-prediction-model/
│
├── codes/
│   ├── check.py
│   ├── dataexplore.py
│   ├── ridge.py
│   ├── lightgbm.py
│   ├── XGBoost.py
│   └── modelcheck.py
│
├── datas/
│   └── cardekho_dataset.csv
│
├── input/
│   └── main.py
│
├── app.py
├── car_price_model.pkl
├── democheckmodel.py
├── requirements.txt
└── .gitignore
