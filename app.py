import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the model and scaler
model_path = 'my_model.h5'
scaler_path = 'scaler.pkl'  # Ensure scaler.pkl is also in the same directory

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Create a function to make predictions
def predict(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return np.argmax(prediction[0])

# Streamlit UI
st.title('Model Prediction App')

st.write("Enter the values for prediction:")

# Example input fields
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)

if st.button('Predict'):
    input_data = [feature_1, feature_2]
    prediction = predict(input_data)
    st.write(f"Prediction: {prediction}")
