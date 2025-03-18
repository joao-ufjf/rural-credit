import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from config import load_models, FEATURE_NAMES
import json

# Set page configuration
st.set_page_config(
    page_title="Rural Credit Prediction",
    page_icon="🌾",
    layout="centered"
)

# Load the models
try:
    scaler, mlp, kmeans = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

def load_cluster_profiles():
    """
    Load cluster profiles from JSON file
    
    Returns:
        dict: Cluster profiles data
    """
    try:
        with open('kmeans_results/cluster_profiles_k4.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Cluster profiles file not found")
        return None

def display_cluster_profile(cluster_num, profiles):
    """
    Display detailed information about a specific cluster profile
    
    Args:
        cluster_num (int): Cluster number (0-based index)
        profiles (dict): Cluster profiles data
    """
    if not profiles:
        return
    
    print(profiles)
    
    profile = profiles.get(str(cluster_num))
    if not profile:
        return
    
    st.subheader(f"Profile Group {cluster_num + 1} Characteristics")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Average Values:**")
        metrics_df = pd.DataFrame({
            'Metric': ['Cost', 'Production', 'Area', 'Credit'],
            'Value': [
                f"R$ {profile['mean_cost']:,.2f}",
                f"R$ {profile['mean_production']:,.2f}",
                f"{profile['mean_area']:.2f} ha",
                f"R$ {profile['mean_credit']:,.2f}"
            ]
        })
        st.table(metrics_df)
    
    with col2:
        st.markdown("**Profile Summary:**")
        st.write(profile['description'])
        
        # Display size and percentage
        st.markdown("**Group Statistics:**")
        st.write(f"- Size: {profile['size']} properties")
        st.write(f"- Represents: {profile['percentage']:.1f}% of total")

def predict_credit(cost, production, area):
    """
    Make prediction using the loaded models
    
    Args:
        cost (float): Cost in Brazilian Reais
        production (float): Production value in Brazilian Reais
        area (float): Area in hectares
        
    Returns:
        tuple: (float: Predicted credit value, int: Cluster assignment)
    """
    # Create input DataFrame with correct feature names
    input_data = pd.DataFrame(
        [[cost, production, area, 0]], 
        columns=FEATURE_NAMES
    )
    
    try:
        scaled_input = scaler.transform(input_data)
        
        # Convert scaled input to DataFrame with feature names (excluding the target column)
        scaled_df = pd.DataFrame(
            scaled_input,
            columns=['cost', 'productivity', 'area', 'value']
        )
        
        # Get cluster assignment
        cluster = kmeans.predict(scaled_df[["cost", "productivity", "value", "area"]])[0]
        
        # Get credit prediction
        prediction = mlp.predict(scaled_df[['cost', 'productivity', 'area']])
        print(prediction)
        
        # Create a dummy array with the predicted value in the last column
        dummy_data = np.zeros((1, len(FEATURE_NAMES)))
        dummy_data[0, -1] = prediction[0]
        
        # Inverse transform to get the original scale
        unscaled_prediction = scaler.inverse_transform(dummy_data)[0, -1]
        return float(unscaled_prediction), int(cluster)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}") from e

def create_float_input(label, help_text, key=None, value=0.0):
    """
    Creates a standardized float number input field
    
    Args:
        label (str): Label for the input field
        help_text (str): Help text to display
        key (str, optional): Unique key for the input field
        value (float, optional): Initial value for the input field
        
    Returns:
        float: Input value
    """
    return st.number_input(
        label,
        min_value=0.0,
        value=value,
        step=0.01,
        format="%.2f",
        help=help_text,
        key=key
    )

def load_sample_data():
    """
    Load and cache sample data from the CSV file
    
    Returns:
        pd.DataFrame: Sample data
    """
    try:
        # Use st.cache_data to avoid reloading the CSV file on every rerun
        @st.cache_data
        def _load_data():
            return pd.read_csv('1_data_to_scale.csv')
        return _load_data()
    except FileNotFoundError:
        st.error("Sample data file not found")
        return None

def generate_random_values():
    """
    Gets a random row from the sample data and makes prediction
    
    Returns:
        tuple: (cost, production, area, prediction, cluster) from real data
    """
    df = load_sample_data()
    if df is None:
        # Fallback to random values if data is not available
        cost = np.random.uniform(1000.0, 100000.0)
        production = np.random.uniform(2000.0, 200000.0)
        area = np.random.uniform(1.0, 100.0)
    else:
        # Select a random row from the dataset
        random_row = df.sample(n=1).iloc[0]
        cost = float(random_row['cost'])
        production = float(random_row['productivity'])
        area = float(random_row['area'])
    
    # Make prediction with the random values
    prediction, cluster = predict_credit(cost, production, area)
    return cost, production, area, prediction, cluster

# Create the web app interface
st.title("🌾 Rural Credit Prediction")
st.write("Enter the following information to predict rural credit:")

# Create input fields
with st.form("prediction_form"):
    # Add random values button above the inputs
    if st.form_submit_button("Generate Random Values", type="secondary"):
        (st.session_state.random_cost, 
         st.session_state.random_production, 
         st.session_state.random_area,
         st.session_state.random_prediction,
         st.session_state.random_cluster) = generate_random_values()
        
        # Display prediction results for random values
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success("Random Values Generated!")
            st.metric(
                label="Predicted Rural Credit",
                value=f"R$ {st.session_state.random_prediction:,.2f}"
            )
            st.info(f"Property Profile: Group {st.session_state.random_cluster + 1}")
        
        with col2:
            st.subheader("Generated Values")
            summary_df = pd.DataFrame({
                'Metric': ['Cost', 'Production Value', 'Area'],
                'Value': [
                    f'R$ {st.session_state.random_cost:,.2f}', 
                    f'R$ {st.session_state.random_production:,.2f}', 
                    f'{st.session_state.random_area:.2f} ha'
                ]
            })
            st.table(summary_df)

    col1, col2 = st.columns(2)
    
    with col1:
        cost = create_float_input(
            "Cost (R$)",
            "Enter the total cost in Brazilian Reais",
            "cost"
        ) if "random_cost" not in st.session_state else create_float_input(
            "Cost (R$)",
            "Enter the total cost in Brazilian Reais",
            "cost",
            value=st.session_state.random_cost
        )
        
        production = create_float_input(
            "Production Value (R$)",
            "Enter the production value in Brazilian Reais",
            "production"
        ) if "random_production" not in st.session_state else create_float_input(
            "Production Value (R$)",
            "Enter the production value in Brazilian Reais",
            "production",
            value=st.session_state.random_production
        )
    
    with col2:
        area = create_float_input(
            "Area (ha)",
            "Enter the area in hectares",
            "area"
        ) if "random_area" not in st.session_state else create_float_input(
            "Area (ha)",
            "Enter the area in hectares",
            "area",
            value=st.session_state.random_area
        )
    
    submit_button = st.form_submit_button("Predict Credit")

# Make prediction when form is submitted
if submit_button:
    if any(value <= 0 for value in (cost, production, area)):
        st.warning("Please enter positive values for all fields.")
    else:
        try:
            prediction, cluster = predict_credit(cost, production, area)
            
            # Create columns for better layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success("Prediction Complete!")
                st.metric(
                    label="Predicted Rural Credit",
                    value=f"R$ {prediction:,.2f}"
                )
                st.info(f"Property Profile: Group {cluster + 1}")
            
            with col2:
                # Display input summary
                st.subheader("Input Summary")
                summary_df = pd.DataFrame({
                    'Metric': ['Cost', 'Production Value', 'Area'],
                    'Value': [
                        f'R$ {cost:,.2f}', 
                        f'R$ {production:,.2f}', 
                        f'{area:.2f} ha'
                    ]
                })
                st.table(summary_df)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Add information footer
st.markdown("---")