import streamlit as st
import numpy as np
import pandas as pd
import os
from annotated_text import annotated_text, annotation
# Load data
data = pd.read_csv("SwiggyZomato/data.txt")

from sklearn.model_selection import train_test_split
x = np.array(data[["Delivery_person_Age",
                   "Delivery_person_Ratings",
                   "distance"]])
y = np.array(data[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.10,
                                                random_state=42)

# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model


if not os.path.isfile("model.h5"):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xtrain, ytrain, batch_size=1, epochs=9)
    model.save("model.h5")
else:
    model = load_model("SwiggyZomato/model.h5")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
# st.text("")
_, col2, _ = st.columns([1, 2, 1])
st.markdown("""
<style>
body {
    background-color: #F5A623;
}
</style>
""", unsafe_allow_html=True)
st.title("Food Delivery Prediction App")
with col2:
    st.image("SwiggyZomato/image.jpg",width=350)
# st.set_background_color("#F5A623")

# def set_bg_hack_url():
#     st.markdown(
#          f"""
#           <style>
#           .stApp {{
#               background: url("https://inc42.com/wp-content/uploads/2022/05/Qcommerce-Story_Feature-Image.jpg");
#               background-size: cover
#           }}
#           </style>
#           """,
#          unsafe_allow_html=True
#      )
# set_bg_hack_url()
# # st.set_page_config(page_title="Food Delivery Time Prediction App", page_icon=":fork_and_knife:", layout="wide")
# st.image('image.jpg',width=200)
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


