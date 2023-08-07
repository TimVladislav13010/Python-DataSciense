import joblib
import streamlit as st
import numpy as np

model = joblib.load("model.joblib")
st.header("123")

season = st.number_input("season")
yr = st.number_input("yr")
mnth = 	st.number_input("mnth")
holiday = 	st.number_input("holiday")
weekday	= st.number_input("weekday")
workingday = 	st.number_input("workingday")
weathersit = st.number_input("weathersit")
temp = st.number_input("temp")
hum = st.number_input("hum")
windspeed = st.number_input("windspeed")


st.text(
    model.predict(
        np.array([
            season, yr, mnth, holiday, weekday, workingday, weathersit, temp, hum, windspeed
        ]).reshape(1, -1)
    )
)

