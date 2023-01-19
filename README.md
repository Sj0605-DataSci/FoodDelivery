# Food Delivery Time Prediction App

## Introduction

This app is a simple web application that predicts the delivery time for food delivery based on various factors such as the age of the delivery person, their ratings, and the distance of delivery.

## Installation

To use this app, you will need to have the following packages installed:
- streamlit
- numpy
- pandas
- sklearn
- keras

## Usage

Use the sliders to input the age, ratings, and distance of delivery and press the "Predict" button to get the predicted delivery time.

## Data

The data used to train the model is a simple dataset consisting of the age, ratings, distance, and delivery time of multiple delivery persons. The data is loaded from a `.txt` file.

## Model

The model used in this app is a simple LSTM neural network created using Keras. The model is trained using the data and then used to predict the delivery time based on the user's input.

## Limitations

Please note that the predictions made by this app are based on a simple dataset and may not be entirely accurate. The app and the model are intended for demonstration purposes only.
