import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

def validate_data(df):
    if 'date' not in df.columns or 'value' not in df.columns:
        return False
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'value'], inplace=True)
    if df['date'].max() - df['date'].min() < pd.Timedelta(days=365*5):
        return False
    return True

def make_forecast(df):
    df = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast, model

def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    st.pyplot(fig)
