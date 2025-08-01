import pandas as pd
import joblib
import streamlit as st
import numpy as np

# Load the pre-trained model from file
model = joblib.load('dump/proj_model.pkl')

# Configure the Streamlit page (title and layout)
st.set_page_config(page_title="Census Income Prediction", layout="wide")

# Inject custom CSS styles for background, fonts, colors, and button
st.markdown(
    """
    <style>
    /* Background image with a semi-transparent dark overlay */
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                          url("https://images.unsplash.com/photo-1603576669240-0562413aee28");
        background-size: cover;
        background-attachment: fixed;
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Styling for the main page title */
    .title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #ffffff;
        text-align: center;
    }

    /* Styling for the subtitle under the title */
    .subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: #dddddd;
        text-align: center;
    }

    /* Styling for the prediction button */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 6px;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    
    /* Styling for the container that holds all input widgets */
    .input-container {
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Display the page title and subtitle with custom styles
st.markdown('<div class="title">Census Income Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict if an individual earns <b>&gt;50K</b> or <b>&lt;=50K</b> based on their profile.</div>', unsafe_allow_html=True)

# Create a container for user input widgets with two columns
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Left column for some inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 35)
        workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Doctorate', '5th-6th'])
        marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    
    # Right column for the remaining inputs
    with col2:
        occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving'])
        relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
        race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        sex = st.selectbox("Sex", ['Male', 'Female'])
        native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England', 'China', 'Cuba'])
    
    st.markdown('</div>', unsafe_allow_html=True)

# When the user clicks the button, prepare inputs and make prediction
if st.button("Predict Income"):
    # Collect inputs into a dataframe for the model
    input_dict = {
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital_status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'hours_per_week': [hours_per_week],
        'native_country': [native_country]
    }

    df_input = pd.DataFrame(input_dict)

    # One-hot encode categorical inputs and align columns with training data
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # Use the loaded model to predict income bracket
    prediction = model.predict(df_input)[0]
    label = ">50K" if prediction == 1 else "<=50K"

    # Display the prediction result to the user
    st.success(f"Predicted Income Bracket: **{label}**")