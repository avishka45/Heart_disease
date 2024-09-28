import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
heart_data = pd.read_csv(r"C:\Users\anjal\Downloads\heart_disease_data.csv")

# Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Logistic Regression Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Function to predict heart disease
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have Heart Disease'

# Custom CSS to set a pastel background and minimal design, keeping everything centered
st.markdown("""
    <style>
    .main {
        background-color: #ffe6e6; /* Soft pastel red background */
        color: #333333; /* Dark text for contrast */
    }
    .stButton>button {
        background-color: #ff9999; /* Light pastel red button */
        color: white;
        border-radius: 5px;
    }
    h1 {
        color: #ff6666;
    }
    .block-container {
        padding: 2rem 4rem;  /* Padding around the content */
        max-width: 700px;     /* Width to keep content centered */
        margin: auto;         /* Center content horizontally */
    }
    .stTextInput input {
        background-color: #fff2f2; /* Pastel input background */
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
def main():
    st.title('Heart Disease Prediction App')

    # Layout: Side-by-side input/output
    col1, col2 = st.columns([2, 1])  # More space for inputs, less for output
    
    with col1:
        st.subheader("Input Patient Data")
        
        # Input fields for the user
        age = st.number_input('Age', min_value=1, max_value=120, value=62)
        sex = st.radio('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        cp = st.slider('Chest Pain Type (0-3)', min_value=0, max_value=3, value=0)
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, value=140)
        chol = st.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=268)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        restecg = st.slider('Resting Electrocardiographic Results (0-2)', min_value=0, max_value=2, value=0)
        thalach = st.slider('Maximum Heart Rate Achieved', min_value=50, max_value=220, value=160)
        exang = st.radio('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        oldpeak = st.slider('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=3.6, step=0.1)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
        ca = st.slider('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0)
        thal = st.selectbox('Thalassemia', options=[1, 2, 3], format_func=lambda x: 'Normal' if x == 1 else ('Fixed Defect' if x == 2 else 'Reversible Defect'))

        # Create a button for prediction
        if st.button('Predict'):
            input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            result = heart_disease_prediction(input_data)
            
            # Show the prediction result
            st.success(result)

# Run the Streamlit app
if __name__ == '__main__':
    main()
