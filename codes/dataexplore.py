import numpy as np
import pandas as pd
import matplotlib as plt 
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("datas/cardekho_dataset.csv")
print(data.columns)

uniquesname=data["brand"].unique()
print(uniquesname)

data = data[data["vehicle_age"] <= 22].copy()
data = data.drop(columns=["seller_type"])

names=data["brand"].value_counts()
print(names)

data = data[data["brand"].map(data["brand"].value_counts()) >= 40]

#after deleting the less them 40 cars brands
print(data["brand"].value_counts())


data["brand_model"] = data["brand"] + "_" + data["model"]
data = data[data["km_driven"] <= 500000]
age_price = data.groupby("vehicle_age")["selling_price"].mean()

plt.figure(figsize=(10, 6))
plt.plot(age_price.index, age_price.values, marker="o")

plt.xlabel("Vehicle Age")
plt.ylabel("Average Selling Price")
plt.title("Vehicle Age vs Average Selling Price")

plt.show()



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Vehicle Age vs Selling Price
age_price = data.groupby("vehicle_age")["selling_price"].mean()

axes[0].plot(age_price.index, age_price.values, marker="o")
axes[0].set_title("Vehicle Age vs Selling Price")
axes[0].set_xlabel("Vehicle Age")
axes[0].set_ylabel("Average Selling Price")


# 2. KM Driven vs Selling Price
km_price = data.groupby("km_driven")["selling_price"].mean()

axes[1].scatter(km_price.index, km_price.values)
axes[1].set_title("KM Driven vs Selling Price")
axes[1].set_xlabel("KM Driven")
axes[1].set_ylabel("Average Selling Price")


# 3. Max Power vs Selling Price
power_price = data.groupby("max_power")["selling_price"].mean()

axes[2].scatter(power_price.index, power_price.values)
axes[2].set_title("Max Power vs Selling Price")
axes[2].set_xlabel("Max Power")
axes[2].set_ylabel("Average Selling Price")

plt.tight_layout()
plt.show()
print(
    data.nlargest(10, "km_driven")[
        ["brand", "model", "vehicle_age", "km_driven", "selling_price"]
    ]
)
print(data["km_driven"].describe())

print(data["km_driven"].quantile([0.90, 0.95, 0.99, 0.995, 0.999]))

print(
    data[data["km_driven"] > 180000][
        ["brand", "model", "vehicle_age", "km_driven", "selling_price"]
    ].sort_values("km_driven", ascending=False).to_string(index=False)
)

print(data[data["km_driven"] > 500000][
    ["brand", "model", "vehicle_age", "km_driven", "selling_price"]
].sort_values("km_driven", ascending=False))


print(data.columns)

plt.figure(figsize=(8, 5))

data.groupby("transmission_type")["selling_price"].mean().plot(
    kind="bar"
)
#here is the table for the average selling price of the cars based on the transmission type. The bar chart shows the average selling price for each transmission type, allowing us to compare the prices between manual and automatic transmissions.
plt.xlabel("Transmission Type")
plt.ylabel("Average Selling Price")
plt.title("Transmission Type vs Average Selling Price")

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


