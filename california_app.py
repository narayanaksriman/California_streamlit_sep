import joblib
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression


@st.cache_resource
def load_model():
    model_path = "california.joblib"
    try:
        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "model" in obj and "columns" in obj:
            return obj["model"], obj["columns"]
    except Exception:
        pass

    data = fetch_california_housing(as_frame=True)
    model = LinearRegression()
    model.fit(data.data, data.target)

    obj = {"model": model, "columns": list(data.data.columns)}
    joblib.dump(obj, model_path)
    return model, obj["columns"]


model, cols = load_model()

st.title("California Housing Price App")

inputs = []
for column in cols:
    value = st.number_input(f"Enter {column} value:", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    prediction = model.predict([inputs])[0]
    st.success(f"The Median House Value is: ${prediction:,.2f}")
