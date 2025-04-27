import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('credit_card_risk.pkl', 'rb'))

# App title
st.title('Credit Risk Prediction App (Logistic Regression)')

# User Inputs
duration = st.slider('Duration in months', 4, 72)
credit_amount = st.slider('Credit Amount', 250, 20000)
age = st.slider('Age', 18, 75)
employment_status = st.selectbox('Employment Status', ['Unemployed', 'Short Term', 'Long Term'])

# Manual encoding (adjust based on your dataset)
employment_dict = {'Unemployed': 0, 'Short Term': 1, 'Long Term': 2}
employment_encoded = employment_dict[employment_status]

# Standard scaling input (important if your model was trained on scaled data)
input_data = np.array([[duration, credit_amount, age, employment_encoded]])
# Optional: Standardize manually if needed, or scale before saving model.

# Predict
prediction = model.predict(input_data)

# Result
st.subheader('Prediction Result:')
st.write('Good Credit Risk' if prediction[0]==1 else 'Bad Credit Risk')
