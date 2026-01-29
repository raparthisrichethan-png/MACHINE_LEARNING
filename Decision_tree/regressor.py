import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Decision Tree Regressor", layout="centered")

st.title("🌳 Decision Tree Regressor")

# Generate regression dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + np.random.randn(100) * 3

# Convert to DataFrame
df = pd.DataFrame({
    "X": X.squeeze(),
    "Y": y
})

# Sidebar controls
st.sidebar.header("⚙️ Model Settings")
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
model.fit(x_train, y_train)

# Input
st.sidebar.header("📌 Input")
x_input = st.sidebar.slider(
    "Enter X value",
    float(X.min()),
    float(X.max()),
        float(X.mean())
)

input_data = np.array([[x_input]])

# Prediction
prediction = model.predict(input_data)[0]
r2 = r2_score(y_test, model.predict(x_test))

# Output
st.success(f"Predicted Output: **{prediction:.2f}**")
st.write(f"📊 R² Score: **{r2:.2f}**")

# View dataset
with st.expander("📂 View Regression Dataset"):
    st.dataframe(df)