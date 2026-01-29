import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

st.title("🏠 California House Price Prediction using Decision Tree Regression")


# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    return df


df = load_data()

# ------------------- PREPROCESSING -------------------
st.subheader("📌 Dataset Preview")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

# Mapping ocean proximity as per notebook
ocean_map = {"<1H OCEAN": 1, "INLAND": 2, "ISLAND": 3, "NEAR OCEAN": 4, "NEAR BAY": 5}

if "ocean_proximity" in df.columns:
    df["ocean_proximity"] = df["ocean_proximity"].map(ocean_map)

# Fill missing values
df = df.fillna(df.median(numeric_only=True))

# ------------------- SPLIT DATA -------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------- SIDEBAR CONTROLS -------------------
st.sidebar.header("⚙️ Model Settings")

max_depth = st.sidebar.slider("max_depth", 1, 50, 10)
min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 20, 1)

use_gridsearch = st.sidebar.checkbox(
    "Use GridSearchCV (Hyperparameter Tuning)", value=False
)

# ------------------- TRAIN MODEL -------------------
if use_gridsearch:
    st.sidebar.write("GridSearchCV Enabled ✅")

    params = {
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    model = DecisionTreeRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid=params, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    st.sidebar.success(f"Best Params: {grid.best_params_}")

else:
    best_model = DecisionTreeRegressor(
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("R2 Score", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")

st.subheader("📈 Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

st.subheader("🧮 Predict House Price")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    longitude = col1.number_input("longitude", value=float(X["longitude"].mean()))
    latitude = col1.number_input("latitude", value=float(X["latitude"].mean()))
    housing_median_age = col1.number_input(
        "housing_median_age", value=float(X["housing_median_age"].mean())
    )

    total_rooms = col2.number_input("total_rooms", value=float(X["total_rooms"].mean()))
    total_bedrooms = col2.number_input(
        "total_bedrooms", value=float(X["total_bedrooms"].mean())
    )
    population = col2.number_input("population", value=float(X["population"].mean()))

    households = col3.number_input("households", value=float(X["households"].mean()))
    median_income = col3.number_input(
        "median_income", value=float(X["median_income"].mean())
    )
    ocean_proximity = col3.selectbox(
        "ocean_proximity",
        options=list(ocean_map.values()),
        format_func=lambda x: [k for k, v in ocean_map.items() if v == x][0],
    )

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame(
        [
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        ]
    )

    prediction = best_model.predict(input_data)[0]
    st.success(f"🏡 Predicted Median House Value: **${prediction:,.2f}**")