import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import time
import argparse
import re

# Import functions from model.py
from model import (
    load_data, prepare_national_trend_data, prepare_state_trend_data, 
    prepare_county_trend_data, train_linear_model, train_random_forest_model,
    train_prophet_model, forecast_linear, forecast_random_forest, forecast_prophet,
    save_model, load_model, get_available_states, get_available_counties, get_year_range
)

# Global variable for models directory
MODELS_DIR = 'models'

# Set page configuration
st.set_page_config(
    page_title="Unemployment Trend Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data
def load_cached_data(data_path='reshaped_unemployment_data.csv'):
    return load_data(data_path)

# Function to create models directory if it doesn't exist
def ensure_model_directory(models_dir=MODELS_DIR):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

# Main app function
def main(data_path='reshaped_unemployment_data.csv', models_dir=MODELS_DIR):
    # Set global models directory
    global MODELS_DIR
    MODELS_DIR = models_dir
    
    # Header
    st.markdown('<div class="main-header">Unemployment Trend Forecaster</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    This application analyzes historical unemployment data and forecasts future trends using machine learning models.
    You can explore trends at national, state, or county levels and predict future unemployment rates.
    </div>
    """, unsafe_allow_html=True)
    
    # Display models directory
    st.markdown(f"<div class='info-text'>Using models from: <code>{models_dir}</code></div>", unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            df = load_cached_data(data_path)
            st.success("Data loaded successfully!")
            
            # Get year range from data
            min_year, max_year = get_year_range(df)
            st.markdown(f"<div class='info-text'>Dataset contains unemployment data from {min_year} to {max_year}.</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the analysis mode",
        ["Data Overview", "National Trends", "State-Level Analysis", "County-Level Analysis"]
    )
    
    # Ensure models directory exists
    ensure_model_directory(models_dir)
    
    # Data Overview
    if app_mode == "Data Overview":
        display_data_overview(df)
    
    # National Trends
    elif app_mode == "National Trends":
        display_national_trends(df, min_year, max_year)
    
    # State-Level Analysis
    elif app_mode == "State-Level Analysis":
        display_state_trends(df, min_year, max_year)
    
    # County-Level Analysis
    elif app_mode == "County-Level Analysis":
        display_county_trends(df, min_year, max_year)

def display_data_overview(df):
    st.markdown('<div class="sub-header">Data Overview</div>', unsafe_allow_html=True)
    
    # Display basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### Dataset Information")
        st.write(f"Total Records: {df.shape[0]:,}")
        st.write(f"Number of States: {df['State'].nunique()}")
        st.write(f"Number of Counties: {df['Area_Name'].nunique()}")
        st.write(f"Year Range: {df['Year'].min()} - {df['Year'].max()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### Unemployment Statistics")
        st.write(f"Average Unemployment Rate: {df['Unemployment_Rate'].mean():.2f}%")
        st.write(f"Minimum Unemployment Rate: {df['Unemployment_Rate'].min():.2f}%")
        st.write(f"Maximum Unemployment Rate: {df['Unemployment_Rate'].max():.2f}%")
        st.write(f"Median Unemployment Rate: {df['Unemployment_Rate'].median():.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample data
    st.markdown("### Sample Data")
    st.dataframe(df.sample(10))
    
    # Distribution of unemployment rates
    st.markdown("### Distribution of Unemployment Rates")
    fig = px.histogram(
        df, x="Unemployment_Rate", 
        nbins=50, 
        title="Distribution of Unemployment Rates",
        labels={"Unemployment_Rate": "Unemployment Rate (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Unemployment rate over time (national average)
    st.markdown("### National Average Unemployment Rate Over Time")
    national_avg = df.groupby('Year')['Unemployment_Rate'].mean().reset_index()
    fig = px.line(
        national_avg, 
        x="Year", 
        y="Unemployment_Rate",
        title="National Average Unemployment Rate Over Time",
        labels={"Unemployment_Rate": "Unemployment Rate (%)", "Year": "Year"}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_national_trends(df, min_year, max_year):
    st.markdown('<div class="sub-header">National Unemployment Trends</div>', unsafe_allow_html=True)
    
    # Prepare national trend data
    national_data = prepare_national_trend_data(df)
    
    # Display historical trend
    st.markdown("### Historical National Unemployment Rate")
    fig = px.line(
        national_data, 
        x="Year", 
        y="Unemployment_Rate",
        title="Historical National Unemployment Rate",
        labels={"Unemployment_Rate": "Unemployment Rate (%)", "Year": "Year"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model selection for forecasting
    st.markdown('<div class="sub-header">Forecast Future Unemployment Rates</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Forecasting Model",
            ["Linear Regression", "Random Forest", "Prophet"]
        )
    
    with col2:
        # Year range for forecasting
        forecast_years = st.slider(
            "Select Year Range for Forecast",
            min_value=max_year + 1,
            max_value=2050,
            value=(2023, 2030)
        )
    
    # Train and forecast
    if st.button("Generate Forecast"):
        with st.spinner("Training model and generating forecast..."):
            # Check if model exists, otherwise train
            if model_type == "Linear Regression":
                model_key = "linear_national"
                model = load_model("linear", "national", models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_linear_model(national_data)
                    save_model(model, "linear", "national", models_dir=MODELS_DIR)
                
                # Generate forecast
                future_years = list(range(forecast_years[0], forecast_years[1] + 1))
                forecast = forecast_linear(model, future_years)
                
                # Plot forecast
                plot_forecast(national_data, forecast, "National Unemployment Rate Forecast (Linear Regression)")
                
            elif model_type == "Random Forest":
                model_key = "rf_national"
                model = load_model("rf", "national", models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_random_forest_model(national_data)
                    save_model(model, "rf", "national", models_dir=MODELS_DIR)
                
                # Generate forecast
                future_years = list(range(forecast_years[0], forecast_years[1] + 1))
                forecast = forecast_random_forest(model, future_years)
                
                # Plot forecast
                plot_forecast(national_data, forecast, "National Unemployment Rate Forecast (Random Forest)")
                
            elif model_type == "Prophet":
                model_key = "prophet_national"
                model = load_model("prophet", "national", models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_prophet_model(national_data)
                    save_model(model, "prophet", "national", models_dir=MODELS_DIR)
                
                # Generate forecast
                periods = forecast_years[1] - max_year
                forecast = forecast_prophet(model, periods)
                
                # Plot forecast with confidence intervals
                plot_prophet_forecast(national_data, forecast, "National Unemployment Rate Forecast (Prophet)")

def display_state_trends(df, min_year, max_year):
    st.markdown('<div class="sub-header">State-Level Unemployment Trends</div>', unsafe_allow_html=True)
    
    # Get available states
    states = get_available_states(df)
    
    # State selection
    selected_state = st.selectbox("Select State", states)
    
    # Prepare state trend data
    state_data = prepare_state_trend_data(df, selected_state)
    
    # Display historical trend
    st.markdown(f"### Historical Unemployment Rate for {selected_state}")
    fig = px.line(
        state_data, 
        x="Year", 
        y="Unemployment_Rate",
        title=f"Historical Unemployment Rate for {selected_state}",
        labels={"Unemployment_Rate": "Unemployment Rate (%)", "Year": "Year"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare with national average
    st.markdown("### Comparison with National Average")
    national_data = prepare_national_trend_data(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=state_data["Year"], 
        y=state_data["Unemployment_Rate"],
        mode='lines',
        name=f'{selected_state}'
    ))
    fig.add_trace(go.Scatter(
        x=national_data["Year"], 
        y=national_data["Unemployment_Rate"],
        mode='lines',
        name='National Average'
    ))
    fig.update_layout(
        title=f"Unemployment Rate: {selected_state} vs. National Average",
        xaxis_title="Year",
        yaxis_title="Unemployment Rate (%)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model selection for forecasting
    st.markdown('<div class="sub-header">Forecast Future Unemployment Rates</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Forecasting Model",
            ["Linear Regression", "Random Forest", "Prophet"],
            key="state_model"
        )
    
    with col2:
        # Year range for forecasting
        forecast_years = st.slider(
            "Select Year Range for Forecast",
            min_value=max_year + 1,
            max_value=2050,
            value=(2023, 2030),
            key="state_years"
        )
    
    # Train and forecast
    if st.button("Generate State Forecast"):
        with st.spinner("Training model and generating forecast..."):
            # Filter data for the selected state
            state_trend = state_data[state_data['State'] == selected_state].copy()
            state_trend = state_trend[['Year', 'Unemployment_Rate']]
            
            # Check if model exists, otherwise train
            if model_type == "Linear Regression":
                model_key = f"linear_state_{selected_state}"
                model = load_model("linear", "state", selected_state, models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_linear_model(state_trend)
                    save_model(model, "linear", "state", selected_state, models_dir=MODELS_DIR)
                
                # Generate forecast
                future_years = list(range(forecast_years[0], forecast_years[1] + 1))
                forecast = forecast_linear(model, future_years)
                
                # Plot forecast
                plot_forecast(state_trend, forecast, f"{selected_state} Unemployment Rate Forecast (Linear Regression)")
                
            elif model_type == "Random Forest":
                model_key = f"rf_state_{selected_state}"
                model = load_model("rf", "state", selected_state, models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_random_forest_model(state_trend)
                    save_model(model, "rf", "state", selected_state, models_dir=MODELS_DIR)
                
                # Generate forecast
                future_years = list(range(forecast_years[0], forecast_years[1] + 1))
                forecast = forecast_random_forest(model, future_years)
                
                # Plot forecast
                plot_forecast(state_trend, forecast, f"{selected_state} Unemployment Rate Forecast (Random Forest)")
                
            elif model_type == "Prophet":
                model_key = f"prophet_state_{selected_state}"
                model = load_model("prophet", "state", selected_state, models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_prophet_model(state_trend)
                    save_model(model, "prophet", "state", selected_state, models_dir=MODELS_DIR)
                
                # Generate forecast
                periods = forecast_years[1] - max_year
                forecast = forecast_prophet(model, periods)
                
                # Plot forecast with confidence intervals
                plot_prophet_forecast(state_trend, forecast, f"{selected_state} Unemployment Rate Forecast (Prophet)")

def display_county_trends(df, min_year, max_year):
    st.markdown('<div class="sub-header">County-Level Unemployment Trends</div>', unsafe_allow_html=True)
    
    # Get available states
    states = get_available_states(df)
    
    # State and county selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_state = st.selectbox("Select State", states, key="county_state")
    
    # Get counties for selected state
    counties = get_available_counties(df, selected_state)
    
    with col2:
        selected_county = st.selectbox("Select County", counties)
    
    # Prepare county trend data
    county_data = prepare_county_trend_data(df, selected_state, selected_county)
    
    # Display historical trend
    st.markdown(f"### Historical Unemployment Rate for {selected_county}")
    fig = px.line(
        county_data, 
        x="Year", 
        y="Unemployment_Rate",
        title=f"Historical Unemployment Rate for {selected_county}",
        labels={"Unemployment_Rate": "Unemployment Rate (%)", "Year": "Year"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare with state and national average
    st.markdown("### Comparison with State and National Average")
    
    # Get state data
    state_data = prepare_state_trend_data(df, selected_state)
    state_data = state_data.groupby('Year')['Unemployment_Rate'].mean().reset_index()
    
    # Get national data
    national_data = prepare_national_trend_data(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=county_data["Year"], 
        y=county_data["Unemployment_Rate"],
        mode='lines',
        name=f'{selected_county}'
    ))
    fig.add_trace(go.Scatter(
        x=state_data["Year"], 
        y=state_data["Unemployment_Rate"],
        mode='lines',
        name=f'{selected_state} Average'
    ))
    fig.add_trace(go.Scatter(
        x=national_data["Year"], 
        y=national_data["Unemployment_Rate"],
        mode='lines',
        name='National Average'
    ))
    fig.update_layout(
        title=f"Unemployment Rate: {selected_county} vs. {selected_state} vs. National Average",
        xaxis_title="Year",
        yaxis_title="Unemployment Rate (%)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model selection for forecasting
    st.markdown('<div class="sub-header">Forecast Future Unemployment Rates</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Forecasting Model",
            ["Linear Regression", "Random Forest", "Prophet"],
            key="county_model"
        )
    
    with col2:
        # Year range for forecasting
        forecast_years = st.slider(
            "Select Year Range for Forecast",
            min_value=max_year + 1,
            max_value=2050,
            value=(2023, 2030),
            key="county_years"
        )
    
    # Train and forecast
    if st.button("Generate County Forecast"):
        with st.spinner("Training model and generating forecast..."):
            # Check if model exists, otherwise train
            # Create a safe identifier for the county
            county_identifier = f"{selected_state}_{selected_county.replace(' ', '_').replace(',', '')}"
            # Sanitize identifier to make it safe for filenames
            county_identifier = re.sub(r'[\\/*?:"<>|]', '_', county_identifier)
            
            if model_type == "Linear Regression":
                model_key = f"linear_county_{county_identifier}"
                model = load_model("linear", "county", county_identifier, models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_linear_model(county_data)
                    save_model(model, "linear", "county", county_identifier, models_dir=MODELS_DIR)
                
                # Generate forecast
                future_years = list(range(forecast_years[0], forecast_years[1] + 1))
                forecast = forecast_linear(model, future_years)
                
                # Plot forecast
                plot_forecast(county_data, forecast, f"{selected_county} Unemployment Rate Forecast (Linear Regression)")
                
            elif model_type == "Random Forest":
                model_key = f"rf_county_{county_identifier}"
                model = load_model("rf", "county", county_identifier, models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_random_forest_model(county_data)
                    save_model(model, "rf", "county", county_identifier, models_dir=MODELS_DIR)
                
                # Generate forecast
                future_years = list(range(forecast_years[0], forecast_years[1] + 1))
                forecast = forecast_random_forest(model, future_years)
                
                # Plot forecast
                plot_forecast(county_data, forecast, f"{selected_county} Unemployment Rate Forecast (Random Forest)")
                
            elif model_type == "Prophet":
                model_key = f"prophet_county_{county_identifier}"
                model = load_model("prophet", "county", county_identifier, models_dir=MODELS_DIR)
                if model is None:
                    st.warning("Model not found in models directory. Training new model...")
                    model = train_prophet_model(county_data)
                    save_model(model, "prophet", "county", county_identifier, models_dir=MODELS_DIR)
                
                # Generate forecast
                periods = forecast_years[1] - max_year
                forecast = forecast_prophet(model, periods)
                
                # Plot forecast with confidence intervals
                plot_prophet_forecast(county_data, forecast, f"{selected_county} Unemployment Rate Forecast (Prophet)")

def plot_forecast(historical_data, forecast_data, title):
    """
    Plot historical data and forecast
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data["Year"], 
        y=historical_data["Unemployment_Rate"],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data["Year"], 
        y=forecast_data["Predicted_Unemployment_Rate"],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Add vertical line to separate historical and forecast data
    max_historical_year = historical_data["Year"].max()
    
    fig.add_shape(
        type="line",
        x0=max_historical_year,
        y0=0,
        x1=max_historical_year,
        y1=max(historical_data["Unemployment_Rate"].max(), forecast_data["Predicted_Unemployment_Rate"].max()) * 1.1,
        line=dict(color="gray", width=2, dash="dot")
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Unemployment Rate (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast data in a table
    st.markdown("### Forecast Data")
    st.dataframe(forecast_data)

def plot_prophet_forecast(historical_data, forecast_data, title):
    """
    Plot historical data and Prophet forecast with confidence intervals
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data["Year"], 
        y=historical_data["Unemployment_Rate"],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data["Year"], 
        y=forecast_data["Predicted_Unemployment_Rate"],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_data["Year"].tolist() + forecast_data["Year"].tolist()[::-1],
        y=forecast_data["Upper_Bound"].tolist() + forecast_data["Lower_Bound"].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='95% Confidence Interval'
    ))
    
    # Add vertical line to separate historical and forecast data
    max_historical_year = historical_data["Year"].max()
    
    fig.add_shape(
        type="line",
        x0=max_historical_year,
        y0=0,
        x1=max_historical_year,
        y1=max(historical_data["Unemployment_Rate"].max(), forecast_data["Upper_Bound"].max()) * 1.1,
        line=dict(color="gray", width=2, dash="dot")
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Unemployment Rate (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast data in a table
    st.markdown("### Forecast Data")
    st.dataframe(forecast_data)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Unemployment Trend Forecaster Streamlit app')
    parser.add_argument('--data', type=str, default='reshaped_unemployment_data.csv',
                        help='Path to the unemployment data CSV file')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    
    # Parse known args to avoid conflicts with Streamlit's own arguments
    args, unknown = parser.parse_known_args()
    
    # Run the app
    main(data_path=args.data, models_dir=args.models_dir) 