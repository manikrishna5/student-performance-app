import streamlit as st
import numpy as np

import os
import joblib

# Correct file paths in Streamlit Cloud
model_path = 'student_performance_app/linear_model.pkl'
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load the saved model and scaler


st.title("Student Exam Score Predictor")

# User inputs
age = st.number_input("Age", min_value=10, max_value=100, value=20)
study_hours = st.number_input("Study hours per day", min_value=0, max_value=24, value=4)
attendance = st.number_input("Attendance percentage", min_value=0, max_value=100, value=80)
sleep_hours = st.number_input("Sleep hours per day", min_value=0, max_value=24, value=7)
mental_health = st.slider("Mental health rating (1-10)", min_value=1, max_value=10, value=7)

social_media_hours = st.number_input("Social Media hours per day", min_value=0, max_value=24, value=2)
netflix_hours = st.number_input("Netflix hours per day", min_value=0, max_value=24, value=1)
extra_wastage_time = social_media_hours + netflix_hours

diet = st.selectbox("Diet Quality", options=["Poor", "Fair", "Good"], index=1)
exercise = st.selectbox("Exercise frequency", options=["None", "Occasionally", "Regularly"], index=1)
part_time_job = st.selectbox("Part-time job", options=["No", "Yes"], index=0)
extracurricular = st.selectbox("Extracurricular participation", options=["No", "Yes"], index=0)
gender = st.selectbox("Gender", options=["Female", "Male", "Other"], index=0)
parental_education = st.selectbox("Parental Education Info Available?", options=["Yes", "No"], index=0)

# Encoding categorical values
diet_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2}
diet = diet_mapping[diet]

exercise_mapping = {'None': 0, 'Occasionally': 1, 'Regularly': 2}
exercise = exercise_mapping[exercise]

part_time_job = 1 if part_time_job == 'Yes' else 0
extracurricular = 1 if extracurricular == 'Yes' else 0
gender_male = 1 if gender == 'Male' else 0
gender_other = 1 if gender == 'Other' else 0
parental_education = 1 if parental_education == 'No' else 0  # same as 'Missing' in your preprocessing

# Prepare input data
scaled_features = np.array([[age, study_hours, attendance, sleep_hours, extra_wastage_time, mental_health]])
scaled_features = scaler.transform(scaled_features)

# Add unscaled features to match training set
unscaled_features = np.array([[part_time_job, diet, exercise, parental_education, 2, extracurricular, gender_male, gender_other]])
input_data_final = np.concatenate([scaled_features, unscaled_features], axis=1)

# Predict
if st.button("Predict Exam Score"):
    prediction = np.clip(model.predict(input_data_final), 0, 100)
    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
