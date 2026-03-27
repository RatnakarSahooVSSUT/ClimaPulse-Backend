import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import threading
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI
import tensorflow as tf
import joblib
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# AQI CALCULATION
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

firebase_config = json.loads(os.environ.get("FIREBASE_KEY"))

cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------------------
# CACHE SYSTEM (KEY FIX 🚀)
# -------------------------------

CACHE_TTL = 60  # seconds

cache_lock = threading.Lock()
cached_data = None
last_fetch_time = 0

def get_latest_data():
    global cached_data, last_fetch_time

    with cache_lock:
        current_time = time.time()

        # ✅ Return cached data if within TTL
        if cached_data is not None and (current_time - last_fetch_time < CACHE_TTL):
            return cached_data

        # 🔥 Fetch from Firebase only when needed
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

        data.reverse()

        if len(data) < 12:
            raise ValueError("Not enough data in Firebase")

        cached_data = np.array(data)
        last_fetch_time = current_time

        return cached_data

# -------------------------------
# FASTAPI INIT
# -------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# MODEL LOAD (ONCE ONLY)
# -------------------------------

model = tf.saved_model.load("model_tf_format")
infer = model.signatures["serving_default"]
scaler = joblib.load("scaler.save")

# -------------------------------
# ROUTES
# -------------------------------

@app.get("/")
def home():
    return {"message": "AQI Prediction API Running"}

@app.get("/predict")
def predict(hours: int = 6):
    try:
        input_data = get_latest_data()

        curr_pm25, curr_pm10, curr_temp, curr_hum = input_data[-1]
        curr_aqi = calculate_aqi(curr_pm25, curr_pm10)

        input_scaled = scaler.transform(input_data)

        preds = []
        current = input_scaled.copy()

        for i in range(hours):
            input_tensor = tf.convert_to_tensor(
                current.reshape(1, 12, 4), dtype=tf.float32
            )

            output = infer(input_tensor)
            pred = list(output.values())[0].numpy()[0]

            pm25, pm10, temp = pred

            next_row = [pm25, pm10, temp, current[-1][3]]

            preds.append(pred)
            current = np.vstack([current[1:], next_row])

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

        final_pred = forecast[-1]

        aqi_change = ((final_pred["aqi"] - curr_aqi) / curr_aqi) * 100 if curr_aqi != 0 else 0
        temp_change = final_pred["temperature"] - curr_temp

        aqi_values = [f["aqi"] for f in forecast]
        temp_values = [f["temperature"] for f in forecast]

        aqi_std = np.std(aqi_values)
        temp_std = np.std(temp_values)

        aqi_conf = max(60, 100 - aqi_std)
        temp_conf = max(60, 100 - temp_std)

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