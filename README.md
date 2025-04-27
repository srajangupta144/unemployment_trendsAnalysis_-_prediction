# Unemployment Trend Forecaster

This application analyzes historical unemployment data and forecasts future trends using machine learning models. It provides insights at national, state, and county levels, allowing users to predict unemployment rates for future years.

## Features

- **Data Overview**: View basic statistics and visualizations of the unemployment dataset
- **National Trends**: Analyze and forecast national unemployment trends
- **State-Level Analysis**: Explore and predict unemployment rates for specific states
- **County-Level Analysis**: Examine and forecast unemployment trends at the county level
- **Multiple Forecasting Models**:
  - Linear Regression: Simple trend-based forecasting
  - Random Forest: More complex pattern recognition
  - Prophet: Time series forecasting with confidence intervals

## Dataset

The application uses the `reshaped_unemployment_data.csv` dataset, which contains unemployment data for counties across the United States. The dataset includes the following columns:

- FIPS_Code: Federal Information Processing Standards code for counties
- State: State abbreviation
- Area_Name: County name
- Rural_Urban_Continuum_Code_2013: Classification code for counties
- Urban_Influence_Code_2013: Classification code for counties
- Metro_2013: Metropolitan status indicator
- Year: Year of the data
- Civilian_Labor_Force: Size of civilian labor force
- Employed: Number of employed individuals
- Unemployment_Rate: Unemployment rate as a percentage


![Screenshot 2025-04-27 135510](https://github.com/user-attachments/assets/ed272d30-27a0-49a4-8b40-6aa5bca3040f)


