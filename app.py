import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
import joblib

model = joblib.load("model/lr_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("📈 My Linear Regression Web App")

st.write("Enter the housing feautures below: ")

MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict Price"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    

    st.success(f"Predicted Price: ${prediction[0] * 100000:.2f}")
