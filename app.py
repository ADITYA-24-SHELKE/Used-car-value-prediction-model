import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "car_price_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "datas",
    "cardekho_dataset.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD BRAND + MODEL DATA
# ============================================================

@st.cache_data
def load_brand_models():

    data = pd.read_csv(DATA_PATH)

    data["brand_model"] = (
        data["brand"].astype(str)
        + "_"
        + data["model"].astype(str)
    )

    return sorted(
        data["brand_model"]
        .dropna()
        .unique()
        .tolist()
    )


# ============================================================
# LOAD MODEL + DATA
# ============================================================

try:

    model = load_model()

    brand_models = load_brand_models()

except Exception as e:

    st.error(
        f"Could not load the model or dataset:\n\n{e}"
    )

    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🚗 Car Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate the resale value of a used car'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# VEHICLE DETAILS
# ============================================================

st.subheader("Vehicle Details")


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    vehicle_age = st.number_input(
        "Vehicle Age (years)",
        min_value=0.0,
        max_value=22.0,
        value=5.0,
        step=1.0
    )

    km_driven = st.number_input(
        "KM Driven",
        min_value=0.0,
        max_value=500000.0,
        value=40000.0,
        step=1000.0
    )

    max_power = st.number_input(
        "Max Power (bhp)",
        min_value=1.0,
        max_value=1000.0,
        value=88.0,
        step=1.0
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    mileage = st.number_input(
        "Mileage (km/l)",
        min_value=1.0,
        max_value=100.0,
        value=20.0,
        step=0.1
    )

    engine = st.number_input(
        "Engine (CC)",
        min_value=100.0,
        max_value=10000.0,
        value=1197.0,
        step=1.0
    )


# ============================================================
# CAR INFORMATION
# ============================================================

st.subheader("Car Information")


brand_model = st.selectbox(
    "Brand & Model",
    options=brand_models
)


transmission = st.selectbox(
    "Transmission",
    options=[
        "Manual",
        "Automatic"
    ]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")


predict_button = st.button(
    "🔍 Predict Selling Price",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    new_car = pd.DataFrame({

        "vehicle_age": [vehicle_age],

        "km_driven": [km_driven],

        "max_power": [max_power],

        "mileage": [mileage],

        "engine": [engine],

        "brand_model": [brand_model],

        "transmission_type": [transmission]

    })


    try:

        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            new_car
        )[0]


        # ----------------------------------------------------
        # DISPLAY SUCCESS MESSAGE
        # ----------------------------------------------------

        st.success(
            "Prediction generated successfully."
        )


        # ----------------------------------------------------
        # DISPLAY PRICE
        # ----------------------------------------------------

        st.subheader(
            "Estimated Selling Price"
        )

        st.markdown(
            f"# ₹ {prediction:,.0f}"
        )


    except Exception as e:

        st.error(
            f"Prediction failed:\n\n{e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "This prediction is an estimate based on the trained "
    "machine learning model and the information provided."
)