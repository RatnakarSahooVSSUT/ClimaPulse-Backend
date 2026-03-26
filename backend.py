import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF logs

import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# -------------------------------
# AQI CALCULATION (CPCB)
# -------------------------------

def aqi_pm25(c):
    table = [
        (0,30,0,50),(31,60,51,100),(61,90,101,200),
        (91,120,201,300),(121,250,301,400),(251,500,401,500)
    ]
    for cl,ch,il,ih in table:
        if cl <= c <= ch:
            return ((ih-il)/(ch-cl))*(c-cl)+il
    return 500

def aqi_pm10(c):
    table = [
        (0,50,0,50),(51,100,51,100),(101,250,101,200),
        (251,350,201,300),(351,430,301,400),(431,600,401,500)
    ]
    for cl,ch,il,ih in table:
        if cl <= c <= ch:
            return ((ih-il)/(ch-cl))*(c-cl)+il
    return 500

def calculate_aqi(pm25, pm10):
    return max(aqi_pm25(pm25), aqi_pm10(pm10))

# -------------------------------
# FIREBASE INIT
# -------------------------------

import os
import json

firebase_config = json.loads(os.environ.get("FIREBASE_KEY"))

cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------------
# FETCH LATEST DATA
# -------------------------------

def get_latest_data():
    docs = db.collection("sensor_data") \
             .order_by("timestamp", direction=firestore.Query.DESCENDING) \
             .limit(12).stream()

    data = []

    for doc in docs:
        d = doc.to_dict()

        data.append([
            d.get("pm25", 0),
            d.get("pm10", 0),
            d.get("temperature", 25),
            d.get("humidity", 50)
        ])

    data.reverse()  # oldest → newest

    if len(data) < 12:
        raise ValueError("Not enough data in Firebase (need 12 readings)")

    return np.array(data)

# -------------------------------
# FASTAPI INIT
# -------------------------------

app = FastAPI()

# Load model & scaler
model = tf.keras.models.load_model("model_tf_format")
scaler = joblib.load("scaler.save")

# -------------------------------
# ROOT API
# -------------------------------

@app.get("/")
def home():
    return {"message": "AQI Prediction API Running"}

# -------------------------------
# PREDICTION API
# -------------------------------

@app.get("/predict")
def predict(hours: int = 6):
    try:
        # -------------------------------
        # GET LATEST DATA
        # -------------------------------
        input_data = get_latest_data()

        # current real values
        curr_pm25, curr_pm10, curr_temp, curr_hum = input_data[-1]
        curr_aqi = calculate_aqi(curr_pm25, curr_pm10)

        # -------------------------------
        # SCALE INPUT
        # -------------------------------
        input_scaled = scaler.transform(input_data)

        preds = []
        current = input_scaled.copy()

        # -------------------------------
        # FUTURE PREDICTION LOOP
        # -------------------------------
        for i in range(hours):
            pred = model.predict(current.reshape(1, 12, 4), verbose=0)[0]

            pm25, pm10, temp = pred

            next_row = [
                pm25,
                pm10,
                temp,
                current[-1][3]  # keep humidity constant
            ]

            preds.append(pred)
            current = np.vstack([current[1:], next_row])

        # -------------------------------
        # INVERSE + AQI
        # -------------------------------
        forecast = []

        for i, p in enumerate(preds):
            dummy = [0, 0, 0, 0]
            dummy[0:3] = p

            inv = scaler.inverse_transform([dummy])[0]

            pm25, pm10, temp = inv[:3]
            aqi = calculate_aqi(pm25, pm10)

            forecast.append({
                "hour": f"+{i+1}h",
                "aqi": round(aqi, 1),
                "temperature": round(temp, 1)
            })

        # -------------------------------
        # SUMMARY CALCULATIONS
        # -------------------------------
        final_pred = forecast[-1]

        # % change
        aqi_change = ((final_pred["aqi"] - curr_aqi) / curr_aqi) * 100
        temp_change = final_pred["temperature"] - curr_temp

        # -------------------------------
        # CONFIDENCE CALCULATION
        # -------------------------------
        aqi_values = [f["aqi"] for f in forecast]
        temp_values = [f["temperature"] for f in forecast]

        aqi_std = np.std(aqi_values)
        temp_std = np.std(temp_values)

        # convert to confidence
        aqi_conf = max(60, 100 - aqi_std)
        temp_conf = max(60, 100 - temp_std)

        # -------------------------------
        # FINAL RESPONSE
        # -------------------------------
        return {
            "current": {
                "aqi": round(curr_aqi, 1),
                "temperature": round(curr_temp, 1)
            },
            "summary": {
                "aqi": final_pred["aqi"],
                "temperature": final_pred["temperature"],
                "aqi_change": round(aqi_change, 1),
                "temp_change": round(temp_change, 1),
                "aqi_confidence": round(aqi_conf, 1),
                "temp_confidence": round(temp_conf, 1)
            },
            "forecast": forecast
        }

    except Exception as e:
        return {"error": str(e)}