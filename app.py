#streamlit-end to end web app for giving input and have predictions
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler

model=tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

##streamlit app - because i dont want html tension
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the 'Geography' feature
geo_encoded=ohe.transform([[geography]]).toarray()
geo_df=pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))
input_df=pd.concat([input_data,geo_df],axis=1)
# Scale the input data
input_df_scaled=scaler.transform(input_df)

#prediction churn
prediction=model.predict(input_df_scaled)
st.write(f'churn probability: {prediction[0][0]:.2f}')
if prediction[0][0]>0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
