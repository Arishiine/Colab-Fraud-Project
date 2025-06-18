
import streamlit as st
import pandas as pd
import joblib
import os

# Load the saved preprocessor and models
@st.cache_resource # Cache the resource to avoid reloading on every rerun
def load_models():
    try:
        preprocessor = joblib.load('saved_models/preprocessor.joblib')
        model_lr = joblib.load('saved_models/logistic_regression_model.joblib')
        model_rf = joblib.load('saved_models/random_forest_model.joblib')
        model_lgb = joblib.load('saved_models/lightgbm_model.joblib')
        st.success("Models and preprocessor loaded successfully.")
        return preprocessor, model_lr, model_rf, model_lgb
    except FileNotFoundError:
        st.error("Error: Saved models or preprocessor not found. Please ensure 'saved_models' directory and its contents exist.")
        return None, None, None, None

preprocessor, model_lr, model_rf, model_lgb = load_models()

# Set up the Streamlit app title and description
st.title("Fraud Detection Application")
st.write("Upload a CSV file with transaction data or enter details manually to predict fraud.")

# Option to upload a CSV file
st.header("Upload Transaction Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        new_data_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(new_data_df)

        # Select model for prediction
        model_choice_upload = st.selectbox(
            "Select Model for Prediction (Uploaded Data):",
            ("Random Forest", "Logistic Regression", "LightGBM")
        )

        if st.button("Predict Fraud (Uploaded Data)"):
            if preprocessor and model_lr and model_rf and model_lgb:
                # Make predictions
                if model_choice_upload == "Logistic Regression":
                    model = model_lr
                elif model_choice_upload == "Random Forest":
                    model = model_rf
                else:
                    model = model_lgb

                try:
                    # Ensure the uploaded data has the same columns as the training data (excluding target)
                    # You might need more robust column handling here depending on your data
                    # For simplicity, assuming column order and names match
                    processed_data = preprocessor.transform(new_data_df)
                    predictions = model.predict(processed_data)
                    probabilities = model.predict_proba(processed_data)[:, 1]

                    results_df = new_data_df.copy()
                    results_df['Predicted_Fraud'] = predictions
                    results_df['Probability_Fraud'] = probabilities

                    st.write("\nPrediction Results (Uploaded Data):")
                    st.dataframe(results_df)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.write("Please ensure the uploaded CSV has the correct format and columns matching the training data.")
            else:
                st.warning("Models not loaded. Please check the 'saved_models' directory.")

# Option to enter data manually (example for a few key features)
st.header("Enter Transaction Details Manually")

# You will need to add input fields for each feature your model expects
# This is a simplified example with only a few features
# Get the list of expected features from the preprocessor
if preprocessor:
    # Assuming the preprocessor's fitted transformers can give us feature names
    # This might require inspecting the fitted preprocessor object based on its type
    # For ColumnTransformer, you can access the columns used by each transformer
    try:
        # This is a simplified way to get feature names, may need adjustment
        # You may need to access the feature names differently based on your specific preprocessor setup
        # For example, if using ColumnTransformer with named transformers, you might need to iterate
        # through the transformers and get their feature names.
        # As a temporary fix to unblock, let's define a dummy numerical_features list
        # assuming the original dataset columns are still available in the environment
        # In a real Streamlit app, you'd need a more robust way to get feature names from the loaded preprocessor
        try:
          # Attempt to get feature names from the preprocessor if it's a ColumnTransformer
          if isinstance(preprocessor, ColumnTransformer):
              input_features = []
              for name, transformer, features in preprocessor.transformers_:
                  if hasattr(transformer, 'get_feature_names_out'):
                      input_features.extend(transformer.get_feature_names_out(features))
                  else:
                      # Fallback if the transformer doesn't have get_feature_names_out
                      input_features.extend(features)
          else:
             st.warning("Preprocessor type not recognized for automatic feature name extraction. Please manually define input fields.")
             input_features = [] # Fallback
        except Exception as e:
             st.warning(f"Could not automatically get feature names from preprocessor: {e}")
             st.write("Please manually define the input fields for your features.")
             input_features = [] # Fallback to empty list if feature names can't be extracted

        manual_input_data = {}
        st.write("Please enter values for the following features:")
        # Create input fields dynamically based on identified features
        for feature in input_features:
            # You would need to add more sophisticated input types based on feature dtype
            manual_input_data[feature] = st.text_input(f"{feature}:", "")

        # Add a button to predict with manual input
        if st.button("Predict Fraud (Manual Input)"):
            if preprocessor and model_lr and model_rf and model_lgb:
                # Convert manual input to a DataFrame
                try:
                    # Convert input values to appropriate types if necessary
                    # This is a basic conversion, you might need more specific handling
                    manual_input_processed = {}
                    for feature, value in manual_input_data.items():
                        # Attempt to infer type or use a default (e.g., string)
                        try:
                            # Try converting to float
                            manual_input_processed[feature] = [float(value)]
                        except ValueError:
                            # If conversion to float fails, keep as string
                            manual_input_processed[feature] = [value]

                    manual_input_df = pd.DataFrame(manual_input_processed)

                    st.write("Manual Input Data:")
                    st.dataframe(manual_input_df)

                    # Select model for prediction
                    model_choice_manual = st.selectbox(
                        "Select Model for Prediction (Manual Input):",
                        ("Random Forest", "Logistic Regression", "LightGBM"),
                        key='manual_model_choice' # Add a unique key
                    )

                    # Make predictions
                    if model_choice_manual == "Logistic Regression":
                        model = model_lr
                    elif model_choice_manual == "Random Forest":
                        model = model_rf
                    else:
                        model = model_lgb

                    try:
                         processed_data_manual = preprocessor.transform(manual_input_df)
                         predictions_manual = model.predict(processed_data_manual)
                         probabilities_manual = model.predict_proba(processed_data_manual)[:, 1]

                         st.write("\nPrediction Result (Manual Input):")
                         st.write(f"Predicted Fraud: {'Yes' if predictions_manual[0] == 1 else 'No'}")
                         st.write(f"Probability of Fraud: {probabilities_manual[0]:.4f}")

                    except Exception as e:
                         st.error(f"Error during prediction: {e}")
                         st.write("Please ensure the manual input data matches the expected format and features.")

                except Exception as e:
                    st.error(f"Error creating DataFrame from manual input: {e}")
            else:
                 st.warning("Models not loaded. Please check the 'saved_models' directory.")

else:
    st.info("Models are being loaded...")

st.markdown("---")
st.write("Developed by Your Name/Team") # Optional: Add your name or team name
