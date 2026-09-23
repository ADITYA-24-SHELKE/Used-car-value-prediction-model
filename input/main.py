import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise Exception(
        f"Could not load model.\n\n{e}"
    )


# ============================================================
# LOAD BRAND + MODEL DATA
# ============================================================

try:

    data = pd.read_csv(DATA_PATH)

    data["brand_model"] = (
        data["brand"].astype(str)
        + "_"
        + data["model"].astype(str)
    )

    brand_models = sorted(
        data["brand_model"]
        .dropna()
        .unique()
        .tolist()
    )

except Exception as e:

    raise Exception(
        f"Could not load dataset.\n\n{e}"
    )


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Car Price Predictor")

root.geometry("720x720")

root.resizable(False, False)

root.configure(
    bg="#f4f6f8"
)


# ============================================================
# VARIABLES
# ============================================================

vehicle_age_var = tk.StringVar()
km_driven_var = tk.StringVar()
max_power_var = tk.StringVar()
mileage_var = tk.StringVar()
engine_var = tk.StringVar()
brand_model_var = tk.StringVar()
transmission_var = tk.StringVar()


# ============================================================
# STYLING
# ============================================================

style = ttk.Style()

style.theme_use("clam")

style.configure(
    "TLabel",
    background="#ffffff",
    font=("Segoe UI", 10)
)

style.configure(
    "Input.TEntry",
    font=("Segoe UI", 10),
    padding=8
)

style.configure(
    "Input.TCombobox",
    font=("Segoe UI", 10),
    padding=7
)

style.configure(
    "Predict.TButton",
    font=("Segoe UI", 11, "bold"),
    padding=11
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg="#f4f6f8"
)

header.pack(
    fill="x",
    pady=(28, 18)
)


title = tk.Label(
    header,
    text="Car Price Predictor",
    bg="#f4f6f8",
    fg="#1f2937",
    font=("Segoe UI", 24, "bold")
)

title.pack()


subtitle = tk.Label(
    header,
    text="Enter the vehicle details to estimate its resale value",
    bg="#f4f6f8",
    fg="#6b7280",
    font=("Segoe UI", 10)
)

subtitle.pack(
    pady=(5, 0)
)


# ============================================================
# MAIN CARD
# ============================================================

card = tk.Frame(
    root,
    bg="#ffffff",
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)

card.pack(
    padx=55,
    fill="both",
    expand=True
)


# ============================================================
# FORM
# ============================================================

form = tk.Frame(
    card,
    bg="#ffffff"
)

form.pack(
    padx=45,
    pady=30
)


def add_label(text, row):

    label = tk.Label(
        form,
        text=text,
        bg="#ffffff",
        fg="#374151",
        font=("Segoe UI", 10)
    )

    label.grid(
        row=row,
        column=0,
        sticky="w",
        pady=8
    )


def add_entry(text, variable, row):

    add_label(text, row)

    entry = ttk.Entry(
        form,
        textvariable=variable,
        style="Input.TEntry",
        width=32
    )

    entry.grid(
        row=row,
        column=1,
        padx=(35, 0),
        pady=8
    )

    return entry


# ============================================================
# NUMERIC INPUTS
# ============================================================

add_entry(
    "Vehicle Age (years)",
    vehicle_age_var,
    0
)

add_entry(
    "KM Driven",
    km_driven_var,
    1
)

add_entry(
    "Max Power (bhp)",
    max_power_var,
    2
)

add_entry(
    "Mileage (km/l)",
    mileage_var,
    3
)

add_entry(
    "Engine (CC)",
    engine_var,
    4
)


# ============================================================
# BRAND + MODEL SEARCH
# ============================================================

add_label(
    "Brand & Model",
    5
)


brand_model_combo = ttk.Combobox(
    form,
    textvariable=brand_model_var,
    values=brand_models,
    style="Input.TCombobox",
    width=30
)

brand_model_combo.grid(
    row=5,
    column=1,
    padx=(35, 0),
    pady=8
)


# ============================================================
# BRAND SEARCH
# ============================================================

def search_brand_model(event=None):

    typed = brand_model_var.get().strip().lower()

    if not typed:

        brand_model_combo["values"] = brand_models

        return

    filtered = [
        item
        for item in brand_models
        if typed in item.lower()
    ]

    brand_model_combo["values"] = filtered

    if filtered:

        brand_model_combo.event_generate(
            "<Down>"
        )


brand_model_combo.bind(
    "<KeyRelease>",
    search_brand_model
)


# ============================================================
# TRANSMISSION
# ============================================================

add_label(
    "Transmission",
    6
)


transmission_combo = ttk.Combobox(
    form,
    textvariable=transmission_var,
    values=[
        "Manual",
        "Automatic"
    ],
    state="readonly",
    style="Input.TCombobox",
    width=30
)

transmission_combo.grid(
    row=6,
    column=1,
    padx=(35, 0),
    pady=8
)


# ============================================================
# RESULT AREA
# ============================================================

result_frame = tk.Frame(
    card,
    bg="#f8fafc",
    highlightthickness=1,
    highlightbackground="#e5e7eb"
)

result_frame.pack(
    padx=45,
    fill="x",
    pady=(0, 15)
)


result_title = tk.Label(
    result_frame,
    text="Estimated Selling Price",
    bg="#f8fafc",
    fg="#6b7280",
    font=("Segoe UI", 10)
)

result_title.pack(
    pady=(14, 3)
)


result_label = tk.Label(
    result_frame,
    text="₹ --",
    bg="#f8fafc",
    fg="#111827",
    font=("Segoe UI", 25, "bold")
)

result_label.pack(
    pady=(0, 14)
)


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict_price():

    try:

        # --------------------------------------------
        # GET INPUT
        # --------------------------------------------

        vehicle_age = float(
            vehicle_age_var.get()
        )

        km_driven = float(
            km_driven_var.get()
        )

        max_power = float(
            max_power_var.get()
        )

        mileage = float(
            mileage_var.get()
        )

        engine = float(
            engine_var.get()
        )

        brand_model = (
            brand_model_var.get()
            .strip()
        )

        transmission = (
            transmission_var.get()
            .strip()
        )


        # --------------------------------------------
        # VALIDATION
        # --------------------------------------------

        if vehicle_age < 0:

            raise ValueError(
                "Vehicle age cannot be negative."
            )

        if km_driven < 0:

            raise ValueError(
                "KM driven cannot be negative."
            )

        if max_power <= 0:

            raise ValueError(
                "Max power must be greater than 0."
            )

        if mileage <= 0:

            raise ValueError(
                "Mileage must be greater than 0."
            )

        if engine <= 0:

            raise ValueError(
                "Engine must be greater than 0."
            )

        if not brand_model:

            raise ValueError(
                "Please select a brand and model."
            )

        if not transmission:

            raise ValueError(
                "Please select transmission."
            )


        # --------------------------------------------
        # CREATE DATAFRAME
        # --------------------------------------------

        new_car = pd.DataFrame({

            "vehicle_age": [vehicle_age],

            "km_driven": [km_driven],

            "max_power": [max_power],

            "mileage": [mileage],

            "engine": [engine],

            "brand_model": [brand_model],

            "transmission_type": [transmission]

        })


        # --------------------------------------------
        # PREDICT
        # --------------------------------------------

        prediction = model.predict(
            new_car
        )[0]


        # --------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------

        result_label.config(
            text=f"₹ {prediction:,.0f}"
        )


    except ValueError as e:

        messagebox.showwarning(
            "Check Input",
            str(e)
        )


    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            f"Something went wrong:\n\n{e}"
        )


# ============================================================
# RESET
# ============================================================

def reset_form():

    vehicle_age_var.set("")

    km_driven_var.set("")

    max_power_var.set("")

    mileage_var.set("")

    engine_var.set("")

    brand_model_var.set("")

    transmission_var.set("")

    brand_model_combo["values"] = brand_models

    result_label.config(
        text="₹ --"
    )


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(
    card,
    bg="#ffffff"
)

button_frame.pack(
    pady=(0, 25)
)


predict_button = ttk.Button(
    button_frame,
    text="Predict Price",
    style="Predict.TButton",
    command=predict_price
)

predict_button.grid(
    row=0,
    column=0,
    padx=6
)


reset_button = ttk.Button(
    button_frame,
    text="Reset",
    command=reset_form
)

reset_button.grid(
    row=0,
    column=1,
    padx=6
)


# ============================================================
# ENTER KEY
# ============================================================

root.bind(
    "<Return>",
    lambda event: predict_price()
)


# ============================================================
# START
# ============================================================

root.mainloop()