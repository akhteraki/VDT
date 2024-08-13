import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the model
model_path = 'models/my_model.h5'
model = load_model(model_path)

# Load the scaler
scaler_path = 'models/scaler.pkl'
scaler = joblib.load(scaler_path)

# Load the original dataset to get the feature names (assuming the same file structure)
df = pd.read_csv('Process_data_0and1_latest.csv')
feature_names = df.drop(columns=['Unnamed: 0', 'Y_variable']).columns

# Create a function to make predictions
def predict(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return np.argmax(prediction[0])

# Streamlit UI
st.title('Model Prediction App')

st.write("Enter the values for prediction:")

# Create input fields for each feature
input_data = []
for feature in feature_names:
    feature_value = st.number_input(f"{feature}", value=0.0)
    input_data.append(feature_value)

if st.button('Predict'):
    prediction = predict(input_data)
    st.write(f"Prediction: {prediction}")
