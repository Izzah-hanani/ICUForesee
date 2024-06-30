import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from math import ceil
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

def set_custom_style():
    st.markdown(
        """
        <style>
        .stMarkdown div[data-testid="stMarkdown"] svg {
            color: #FFFFF0 !important; /* Set your desired color */
        }
        .main {
            background-color: #008B8B; /* Set your desired background color */
        }
        body {
            color: #FFFFF0; /* Set text color */
            background-color: #E0FFFF;
            font-family: Arial, sans-serif; /* Set font */
        }
        .stApp {
            max-width: 800px; /* Adjust maximum width of the app */
            margin: auto; /* Center the app */
        }
        .css-18e3th9 {
            padding: 1rem; /* Adjust padding around the content */
        }
        .stNumberInput label {
            color: #FFFFF0; /* Set the label color */
        }
        .stRadio label {
            color: #FFFFF0; /* Set the label color */
        }
        .stNumberInput input {
            background-color: #f0f2f6; /* Set the input background color */
            border: 2px solid #B0E0E6; /* Set the input border color */
        }
        .stNumberInput input:hover {
            border-color: #191970; /* Set the input border color on hover */
        }
        .custom-input {
            background-color: #e0ffff !important; /* Custom background color for input */
            border: 2px solid #7B68EE !important; /* Custom border color for input */
        }
        .custom-label {
            color: #FFFFF0 !important; /* Custom color for labels */
        }
        .stDateInput label {
            color: #FFFFF0; /* Set your desired color */
        }
        .output-box {
            background-color: rgba(255, 255, 255, 0.9); /* Transparent background */
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-top: 20px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        }
        .link-button {
            display: inline-block;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            outline: none;
            color: #fff;
            background-color: #FFFFF0;
            border: none;
            border-radius: 15px;
            box-shadow: 0 9px #999;
            margin-top: 10px; /* Adjust top margin */
        }
        .link-button:hover {background-color: #3e8e41}
        .link-button:active {
            background-color: #3e8e41;
            box-shadow: 0 5px #666;
            transform: translateY(4px);
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column; /* Align children vertically */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# Load your preprocessed data (assuming it's saved as 'cleaned_df.csv')
df = pd.read_csv('cleaned_df.csv')

# Separate features and target variable from the DataFrame
features = ['rcount', 'psychologicaldisordermajor', 'hemo', 'irondef', 'pneum', 'respiration', 'bloodureanitro', 'substancedependence','asthma',
            'psychother', 'malnutrition', 'neutrophils', 'hematocrit', 'bmi', 'sodium', 'glucose', 'dialysisrenalendstage', 'depress', 'fibrosisandother',
            'creatinine', 'pulse']
X = df[features]
y = df['lengthofstay']

model = joblib.load('xgb_model.joblib')

button_link_html = """
    <button onclick="window.open('https://a188471.wixsite.com/icuforesee', '_blank')" class="link-button" style="margin: 10px 20px 15px 1px;">Go Back</button>
    """
components.html(button_link_html)

# Streamlit app
st.title("Length of ICU Stay Prediction")
# Center content
with st.container():
    st.markdown('<div class="centered">', unsafe_allow_html=True)

    # Input fields for features
    rcount = st.number_input("Number of Readmission Patient to ICU", min_value=0, key='rcount', help="Number of times the patient was readmitted")

    # Psychological Disorder Major
    psychologicaldisordermajor = st.radio(
        "Psychological Disorder Major", 
        (0, 1), 
        index=0, 
        help="Presence of a major psychological disorder during encounter (0 - None, 1 - Presence)"
    )
    # Iron Deficiency
    iron_def = st.radio(
        "Iron Deficiency", 
        (0, 1), 
        index=0, 
        help="Presence of iron deficiency during encounter (0 - None, 1 - Presence)"
    )
    pneum = st.radio(
        "Pneumonia", 
        (0, 1), 
        index=0, 
        help="Presence of Pneumonia during encounter (0 - None, 1 - Presence)"
    )
    substancedependence =st.radio(
        "Substance Dependence", 
        (0, 1), 
        index=0, 
        help="Presence of Substance Dependence during encounter (0 - None, 1 - Presence)"
    )
    asthma = st.radio(
        "Asthma", 
        (0, 1), 
        index=0, 
        help="Presence of Asthma during encounter (0 - None, 1 - Presence)"
    )
    psychother = st.radio(
        "Other Psychological Disorder", 
        (0, 1), 
        index=0, 
        help="Presence of Other Psychological Disorder during ecounter (0 - None, 1 - Presence)"
    )
    malnutrition = st.radio(
        "Malnutrition", 
        (0, 1), 
        index=0, 
        help="Presence of Malnutrition during encounter (0 - None, 1 - Presence)"
    )
    depress = st.radio(
        "Depression", 
        (0, 1), 
        index=0, 
        help="Presence of Depression during encounter (0 - None, 1 - Presence)"
    )
    fibrosisandother = st.radio(
        "Fibrosis", 
        (0, 1), 
        index=0, 
        help="Presence of Fibrosis during encounter (0 - None, 1 - Presence)"
    )
    dialysisrenalendstage = st.radio(
        "Renal Disease", 
        (0, 1), 
        index=0, 
        help="Presence of Renal Disease during encounter (0 - None, 1 - Presence)"
    )
    hemo = st.radio(
        "Blood Disorder", 
        (0, 1), 
        index=0, 
        help="Presence of Blood Disorder during encounter (0 - None, 1 - Presence)"
    )
    pulse = st.number_input("Average Pulse Rate", min_value=0)
    respiration = st.number_input("Average Respiration", min_value=0.0)
    bloodureanitro = st.number_input("Average Blood Urea Nitrogen", min_value=0.0)
    neutrophils = st.number_input("Average Neutrophils Value", min_value=0.0)
    hematocrit = st.number_input("Average Hematocrit Value", min_value=0.0)
    bmi = st.number_input("Average Bmi", min_value=0.0)
    sodium = st.number_input("Average Sodium Value", min_value=0.0)
    glucose = st.number_input("Average Glucose Value", min_value=0.0)
    creatinine = st.number_input("Average Creatinine Value", min_value=0.0)

    # Input field for admission start date
    start_date = st.date_input("Admission Start Date")

    # Initialize variables to store prediction results
    rounded_prediction = None
    discharge_date = None
    rmse_all = None

    # Predict button
    if st.button("Predict"):
        # Create a DataFrame for the input features
        input_data = pd.DataFrame([[rcount, psychologicaldisordermajor, hemo, iron_def, pneum, respiration, bloodureanitro, substancedependence,asthma,
                psychother, malnutrition, neutrophils, hematocrit, bmi, sodium, glucose, dialysisrenalendstage, depress, fibrosisandother,
                creatinine, pulse]], columns=features)
        input_data_scaled = input_data
        
        # Predict length of stay
        prediction = model.predict(input_data_scaled)[0]

        # Round prediction to nearest integer
        rounded_prediction = ceil(prediction)
        
        # Calculate discharge date
        discharge_date = start_date + timedelta(days=rounded_prediction)

        # Update the original DataFrame with the new patient data
        new_patient_df = input_data.copy()
        new_patient_df['lengthofstay'] = rounded_prediction
        combined_df = pd.concat([df, new_patient_df], ignore_index=True)

        # Separate features and target variable from the combined DataFrame
        X_combined = combined_df[features]
        y_combined = combined_df['lengthofstay']

        # Recalculate RMSE with the new patient data
        rmse_all = np.sqrt(mean_squared_error(y_combined, model.predict(X_combined)))

    # Display the prediction results
    if rounded_prediction is not None and discharge_date is not None:
       st.markdown(
        f"""
        <div class="output-box">
            <h3>Predicted Length of Stay: {rounded_prediction} days</h3>
            <p>Expected Discharge Date: {discharge_date.strftime('%Y-%m-%d')}</p>
            <p>Root Mean Squared Error (RMSE) including new patient data: {rmse_all:.6f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
