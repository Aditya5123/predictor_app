import streamlit as st
import pandas as pd
import os
from prophet import Prophet
from utils import validate_data, make_forecast, plot_forecast

UPLOAD_DIR = "uploads"
FORECAST_DIR = "forecasts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)

st.title("📈 Personalized Time Series Forecaster (Prophet)")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV with 'date' and 'value' columns", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    filename = os.path.basename(uploaded_file.name)
    file_path = os.path.join(UPLOAD_DIR, filename)
    df.to_csv(file_path, index=False)

    # Validate data
    if not validate_data(df):
        st.error("Invalid file. Make sure it has 'date' and 'value' columns and covers at least 5 years.")
    else:
        st.success("Data looks good! Showing historical plot:")
        df['date'] = pd.to_datetime(df['date'])
        st.line_chart(df.set_index('date')['value'])

        if st.button("Generate 1-Year Forecast"):
            forecast_df, model = make_forecast(df)
            forecast_file = os.path.join(FORECAST_DIR, f"{filename}_forecast.csv")
            forecast_df.to_csv(forecast_file, index=False)
            st.success("Forecast generated!")

            # Plot forecast
            plot_forecast(model, forecast_df)

# Option to load saved files
st.sidebar.header("📂 Load Saved Forecasts")
saved_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv')]
selected = st.sidebar.selectbox("Choose a saved file", saved_files)

if selected:
    df = pd.read_csv(os.path.join(UPLOAD_DIR, selected))
    if validate_data(df):
        st.sidebar.write("Re-plotting forecast...")
        forecast_df, model = make_forecast(df)
        plot_forecast(model, forecast_df)
