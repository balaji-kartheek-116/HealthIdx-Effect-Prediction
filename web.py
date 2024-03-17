import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import pickle

# Load the Linear Regression Model
with open('models/linear_regression_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

# Load the dataset
dataset_path = 'DeviceUsageDuration.csv'
data = pd.read_csv(dataset_path)

# Extract min and max values from the dataset
min_values = data.min()
max_values = data.max()

# Create a StandardScaler and fit it to the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['HealthIndex']))

# Define correct username and password
CORRECT_USERNAME = "admin"
CORRECT_PASSWORD = "password"  # Change this to your desired password

# Streamlit App
def main():
    st.title("Health Index Prediction App")

    # Display the Health.jpg image below the title
    health_image = Image.open('Health.jpg')
    st.image(health_image, caption='Health Image', use_column_width=True)

    # Check if the user is authenticated
    session_state = st.session_state
    if "authenticated" not in session_state:
        session_state.authenticated = False

    # If not authenticated, show the login form
    if not session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
                session_state.authenticated = True
            else:
                st.error("Incorrect username or password")
    
    # If authenticated, show the prediction form and logout button
    if session_state.authenticated:
        st.subheader("Enter Feature Values:")
        age = st.slider("Age", min_value=float(min_values.loc['Age']), max_value=float(max_values.loc['Age']), value=float(min_values.loc['Age']))
        income = st.slider("Income", min_value=float(min_values.loc['Income']), max_value=float(max_values.loc['Income']), value=float(min_values.loc['Income']))
        social_media_spent = st.slider("Social Media Spent", min_value=float(min_values.loc['SocialMediaSpent']), max_value=float(max_values.loc['SocialMediaSpent']), value=float(min_values.loc['SocialMediaSpent']))
        entertainment_spend = st.slider("Entertainment Spend", min_value=float(min_values.loc['EntertainmentSpend']), max_value=float(max_values.loc['EntertainmentSpend']), value=float(min_values.loc['EntertainmentSpend']))
        stress_level = st.slider("Stress Level", min_value=float(min_values.loc['StressLevel']), max_value=float(max_values.loc['StressLevel']), value=float(min_values.loc['StressLevel']))
        usage_duration_minutes = st.slider("Usage Duration (Minutes)", min_value=float(min_values.loc['UsageDurationMinutes']), max_value=float(max_values.loc['UsageDurationMinutes']), value=float(min_values.loc['UsageDurationMinutes']))

        # Make predictions using the Linear Regression model
        if st.button("Predict"):
            # Standardize the user input
            user_input = [[age, income, social_media_spent, entertainment_spend, stress_level, usage_duration_minutes]]
            user_input_scaled = scaler.transform(user_input)

            # Make predictions using the Linear Regression model
            y_pred = lr_model.predict(user_input_scaled)

            # Display the predicted Health Index
            st.subheader("Predicted Health Index:")
            st.write(y_pred[0])

        # Add a logout button
        if st.button("Logout"):
            session_state.authenticated = False

if __name__ == "__main__":
    main()
