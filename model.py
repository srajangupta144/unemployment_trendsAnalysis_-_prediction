import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import joblib
import os
import re
from datetime import datetime
import argparse

def load_data(file_path='reshaped_unemployment_data.csv'):
    """
    Load and preprocess the unemployment dataset
    """
    df = pd.read_csv(file_path)
    return df

def prepare_national_trend_data(df):
    """
    Prepare data for national unemployment trend analysis
    """
    # Group by year and calculate average unemployment rate
    national_trend = df.groupby('Year')['Unemployment_Rate'].mean().reset_index()
    national_trend.columns = ['Year', 'Unemployment_Rate']
    return national_trend

def prepare_state_trend_data(df, state=None):
    """
    Prepare data for state-level unemployment trend analysis
    """
    if state:
        state_data = df[df['State'] == state]
    else:
        state_data = df
        
    # Group by state and year
    state_trend = state_data.groupby(['State', 'Year'])['Unemployment_Rate'].mean().reset_index()
    return state_trend

def prepare_county_trend_data(df, state=None, county=None):
    """
    Prepare data for county-level unemployment trend analysis
    """
    if state:
        county_data = df[df['State'] == state]
        if county:
            county_data = county_data[county_data['Area_Name'] == county]
    else:
        county_data = df
        
    # Group by county and year if needed
    if not county:
        county_trend = county_data.groupby(['State', 'Area_Name', 'Year'])['Unemployment_Rate'].mean().reset_index()
    else:
        county_trend = county_data.groupby(['Year'])['Unemployment_Rate'].mean().reset_index()
        
    return county_trend

def train_linear_model(data):
    """
    Train a linear regression model for forecasting
    """
    X = data[['Year']].values
    y = data['Unemployment_Rate'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def train_random_forest_model(data):
    """
    Train a random forest model for forecasting
    """
    X = data[['Year']].values
    y = data['Unemployment_Rate'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def train_prophet_model(data):
    """
    Train a Prophet model for forecasting
    """
    # Prepare data for Prophet
    prophet_data = data.rename(columns={'Year': 'ds', 'Unemployment_Rate': 'y'})
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')
    
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_data)
    
    return model

def forecast_linear(model, years):
    """
    Generate forecasts using linear regression model
    """
    future_years = np.array(years).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    forecast_df = pd.DataFrame({
        'Year': years,
        'Predicted_Unemployment_Rate': predictions
    })
    
    return forecast_df

def forecast_random_forest(model, years):
    """
    Generate forecasts using random forest model
    """
    future_years = np.array(years).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    forecast_df = pd.DataFrame({
        'Year': years,
        'Predicted_Unemployment_Rate': predictions
    })
    
    return forecast_df

def forecast_prophet(model, periods):
    """
    Generate forecasts using Prophet model
    """
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    
    # Convert forecast to desired format
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    forecast_df = forecast_df.rename(columns={
        'ds': 'Year',
        'yhat': 'Predicted_Unemployment_Rate',
        'yhat_lower': 'Lower_Bound',
        'yhat_upper': 'Upper_Bound'
    })
    
    # Convert datetime to year
    forecast_df['Year'] = forecast_df['Year'].dt.year
    
    return forecast_df

def save_model(model, model_type, level, identifier=None, models_dir='models'):
    """
    Save trained model to disk
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Sanitize identifier to make it safe for filenames
    if identifier:
        # Replace any characters that are problematic in filenames
        identifier = re.sub(r'[\\/*?:"<>|]', '_', identifier)
    
    filename = f"{models_dir}/{model_type}_{level}"
    if identifier:
        filename += f"_{identifier}"
    filename += ".joblib"
    
    joblib.dump(model, filename)
    return filename

def load_model(model_type, level, identifier=None, models_dir='models'):
    """
    Load trained model from disk
    """
    # Sanitize identifier to make it safe for filenames
    if identifier:
        # Replace any characters that are problematic in filenames
        identifier = re.sub(r'[\\/*?:"<>|]', '_', identifier)
    
    filename = f"{models_dir}/{model_type}_{level}"
    if identifier:
        filename += f"_{identifier}"
    filename += ".joblib"
    
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

def get_available_states(df):
    """
    Get list of available states in the dataset
    """
    return sorted(df['State'].unique())

def get_available_counties(df, state):
    """
    Get list of available counties for a given state
    """
    counties = df[df['State'] == state]['Area_Name'].unique()
    return sorted(counties)

def get_year_range(df):
    """
    Get the range of years in the dataset
    """
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    return min_year, max_year

def train_all_models(data_path='reshaped_unemployment_data.csv', models_dir='models', states_limit=None, counties_limit=None):
    """
    Train and save all models (national, state, and county levels)
    
    Parameters:
    - data_path: Path to the unemployment data CSV file
    - models_dir: Directory to save trained models
    - states_limit: Limit training to these states (for testing)
    - counties_limit: Limit training to these counties per state (for testing)
    
    Returns:
    - Dictionary with paths to all trained models
    """
    print("Loading data...")
    df = load_data(data_path)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_paths = {}
    
    # Train national models
    print("Training national models...")
    national_data = prepare_national_trend_data(df)
    
    # Linear Regression
    print("- Training national linear regression model")
    linear_model = train_linear_model(national_data)
    linear_path = save_model(linear_model, "linear", "national", models_dir=models_dir)
    model_paths["national_linear"] = linear_path
    
    # Random Forest
    print("- Training national random forest model")
    rf_model = train_random_forest_model(national_data)
    rf_path = save_model(rf_model, "rf", "national", models_dir=models_dir)
    model_paths["national_rf"] = rf_path
    
    # Prophet
    print("- Training national prophet model")
    prophet_model = train_prophet_model(national_data)
    prophet_path = save_model(prophet_model, "prophet", "national", models_dir=models_dir)
    model_paths["national_prophet"] = prophet_path
    
    # Train state models
    print("Training state models...")
    states = get_available_states(df)
    
    # Limit states if specified
    if states_limit is not None:
        if isinstance(states_limit, int):
            states = states[:min(states_limit, len(states))]
        else:
            states = [state for state in states if state in states_limit]
    
    for state in states:
        print(f"- Processing state: {state}")
        state_data = prepare_state_trend_data(df, state)
        state_trend = state_data[state_data['State'] == state].copy()
        state_trend = state_trend[['Year', 'Unemployment_Rate']]
        
        # Linear Regression
        linear_model = train_linear_model(state_trend)
        linear_path = save_model(linear_model, "linear", "state", state, models_dir=models_dir)
        model_paths[f"state_{state}_linear"] = linear_path
        
        # Random Forest
        rf_model = train_random_forest_model(state_trend)
        rf_path = save_model(rf_model, "rf", "state", state, models_dir=models_dir)
        model_paths[f"state_{state}_rf"] = rf_path
        
        # Prophet
        prophet_model = train_prophet_model(state_trend)
        prophet_path = save_model(prophet_model, "prophet", "state", state, models_dir=models_dir)
        model_paths[f"state_{state}_prophet"] = prophet_path
        
        # Train county models for this state
        counties = get_available_counties(df, state)
        
        # Limit counties if specified
        if counties_limit is not None:
            counties = counties[:min(counties_limit, len(counties))]
        
        for county in counties:
            county_identifier = f"{state}_{county.replace(' ', '_').replace(',', '')}"
            print(f"  - Processing county: {county}")
            
            county_data = prepare_county_trend_data(df, state, county)
            
            # Linear Regression
            linear_model = train_linear_model(county_data)
            linear_path = save_model(linear_model, "linear", "county", county_identifier, models_dir=models_dir)
            model_paths[f"county_{county_identifier}_linear"] = linear_path
            
            # Random Forest
            rf_model = train_random_forest_model(county_data)
            rf_path = save_model(rf_model, "rf", "county", county_identifier, models_dir=models_dir)
            model_paths[f"county_{county_identifier}_rf"] = rf_path
            
            # Prophet
            prophet_model = train_prophet_model(county_data)
            prophet_path = save_model(prophet_model, "prophet", "county", county_identifier, models_dir=models_dir)
            model_paths[f"county_{county_identifier}_prophet"] = prophet_path
    
    print(f"All models trained and saved to {models_dir}/")
    return model_paths

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train unemployment forecasting models')
    parser.add_argument('--data', type=str, default='reshaped_unemployment_data.csv',
                        help='Path to the unemployment data CSV file')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--states-limit', type=int, default=None,
                        help='Limit training to this number of states (for testing)')
    parser.add_argument('--counties-limit', type=int, default=None,
                        help='Limit training to this number of counties per state (for testing)')
    
    args = parser.parse_args()
    
    # Train all models
    train_all_models(
        data_path=args.data,
        models_dir=args.models_dir,
        states_limit=args.states_limit,
        counties_limit=args.counties_limit
    ) 