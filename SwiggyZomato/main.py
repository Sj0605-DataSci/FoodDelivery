import streamlit as st
import numpy as np
import pandas as pd
import os

# Load data
data = pd.read_csv("SwiggyZomato/data.txt")


from keras.models import load_model


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("SwiggyZomato/style/style.css")

model = load_model("SwiggyZomato/model.h5")

_, col2, _ = st.columns([1, 2, 1])

st.title("Food Delivery Prediction App")
with col2:
    st.image("SwiggyZomato/image.jpg",width=350)

st.markdown("### This app uses a predictive model to estimate the delivery time for your food order.")
st.write("Please input the following information to predict the delivery time:")

# Use widgets to get user input
age = st.sidebar.slider("Age of Delivery Person", 18, 60, 30)
ratings = st.sidebar.slider("Ratings of Delivery Person", 1, 5, 3)
distance = st.sidebar.slider("Distance (in Km)", 1, 100, 5)
# Create a predict button
if st.button("Predict"):
    delivery_info = np.array([[age, ratings, distance]])
    predicted_time = model.predict(delivery_info)
    st.write("The predicted delivery time is: ", predicted_time[0][0], " minutes")

st.write("Dataset used: ")
st.dataframe(data.head())

# Add a disclaimer
st.write("Note: The model is trained on the given dataset and may not be accurate for all scenarios.")


