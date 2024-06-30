import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import streamlit as st

# Function to fetch weather data
def fetch_weather_data(api_key, city, start_date, end_date):
    url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
    weather_data = []

    for single_date in pd.date_range(start_date, end_date):
        params = {
            'appid': api_key,
            'lat': city['lat'],
            'lon': city['lon'],
            'dt': int(single_date.timestamp())
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            weather_data.append(response.json())
        else:
            st.error(f"Failed to fetch data for {single_date}")

    return weather_data

# Function to prepare data for modeling
def prepare_data(weather_data):
    df = pd.json_normalize(weather_data, 'hourly')
    df['date'] = pd.to_datetime(df['dt'], unit='s')
    heatwave_threshold = 35  # Example threshold in Celsius
    df['is_heatwave'] = (df['temp'] - 273.15) > heatwave_threshold
    return df

# Function to train model
def train_model(df):
    features = df[['temp', 'humidity', 'pressure', 'wind_speed']]
    labels = df['is_heatwave']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
    joblib.dump(model, 'heatwave_model.pkl')
    return model

# Function to load model
def load_model():
    return joblib.load('heatwave_model.pkl')

# Streamlit app
st.title("Summer Heat Waves Mobile Alert System")

# API key input
api_key = st.text_input("Enter your OpenWeatherMap API key")

# City input
city_name = st.text_input("Enter city name (e.g., New York)")
city_lat = st.number_input("Enter city latitude", value=40.7128)
city_lon = st.number_input("Enter city longitude", value=-74.0060)
city = {'lat': city_lat, 'lon': city_lon}

# Date range input
start_date = st.date_input("Start date", datetime.date(2023, 6, 1))
end_date = st.date_input("End date", datetime.date(2023, 9, 1))

# Fetch and display weather data
if st.button("Fetch Weather Data"):
    weather_data = fetch_weather_data(api_key, city, start_date, end_date)
    if weather_data:
        df = prepare_data(weather_data)
        st.write(df)

        # Plot temperature over time
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['temp'] - 273.15)
        plt.xlabel('Date')
        plt.ylabel('Temperature (C)')
        plt.title('Temperature Over Time')
        st.pyplot(plt)

        # Train and save model
        if st.button("Train Model"):
            model = train_model(df)
            st.success("Model trained and saved successfully!")

# Prediction
st.header("Predict Heatwave")
temp = st.number_input("Temperature (Kelvin)")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")
wind_speed = st.number_input("Wind Speed")

if st.button("Predict Heatwave"):
    model = load_model()
    features = [[temp, humidity, pressure, wind_speed]]
    prediction = model.predict(features)
    if prediction[0]:
        st.warning("Heatwave Alert!")
    else:
        st.success("No Heatwave")
