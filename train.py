import pandas as pd
import glob

TRAIN_MODEL = False  # set True only when training
# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
files = glob.glob("pollution_data/*.csv")

print("Total files:", len(files))

df_list = []

# Load limited files (safe for memory)
files = files[:20]

for file in files:
    try:
        temp = pd.read_csv(
            file,
            usecols=[
                'From Date',
                'PM2.5 (ug/m3)',
                'PM10 (ug/m3)',
                'AT (degree C)',
                'RH (%)'
            ]
        )
        df_list.append(temp)
    except Exception as e:
        print(f"Skipped {file}: {e}")

# Combine all
df = pd.concat(df_list, ignore_index=True)

# -------------------------------
# STEP 2: RENAME COLUMNS
# -------------------------------
df.rename(columns={
    'PM2.5 (ug/m3)': 'pm25',
    'PM10 (ug/m3)': 'pm10',
    'AT (degree C)': 'temperature',
    'RH (%)': 'humidity'
}, inplace=True)

# -------------------------------
# STEP 3: CLEAN DATA
# -------------------------------
df['From Date'] = pd.to_datetime(df['From Date'])
df.sort_values('From Date', inplace=True)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("After cleaning rows:", len(df))

# -------------------------------
# STEP 4: AQI CALCULATION (CPCB)
# -------------------------------

def aqi_pm25(c):
    table = [
        (0,30,0,50),
        (31,60,51,100),
        (61,90,101,200),
        (91,120,201,300),
        (121,250,301,400),
        (251,500,401,500)
    ]
    for cl, ch, il, ih in table:
        if cl <= c <= ch:
            return ((ih-il)/(ch-cl))*(c-cl)+il
    return 500


def aqi_pm10(c):
    table = [
        (0,50,0,50),
        (51,100,51,100),
        (101,250,101,200),
        (251,350,201,300),
        (351,430,301,400),
        (431,600,401,500)
    ]
    for cl, ch, il, ih in table:
        if cl <= c <= ch:
            return ((ih-il)/(ch-cl))*(c-cl)+il
    return 500


def calculate_aqi(pm25, pm10):
    return max(aqi_pm25(pm25), aqi_pm10(pm10))


# Apply AQI
df['aqi'] = df.apply(lambda x: calculate_aqi(x['pm25'], x['pm10']), axis=1)


# -------------------------------
# STEP 5: FINAL CHECK
# -------------------------------
print(df.head())
print("\nNull values:\n", df.isnull().sum())

from sklearn.preprocessing import MinMaxScaler
import joblib

# -------------------------------
# STEP 6: NORMALIZATION
# -------------------------------

# Select features (IMPORTANT ORDER)
features = ['pm25', 'pm10', 'temperature', 'humidity']

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df[features])

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=features)

print("\nScaled Data Sample:")
print(scaled_df.head())

# Save scaler for backend use
joblib.dump(scaler, "scaler.save")

import numpy as np

# -------------------------------
# STEP 7: SEQUENCE CREATION
# -------------------------------

SEQ_LEN = 12  # past 12 readings

def create_sequences(data, seq_len=12):
    X, y = [], []
    
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][[0,1,2]])  # pm25, pm10, temperature
    
    return np.array(X), np.array(y)


# Convert DataFrame to numpy
data_array = scaled_df.values

X, y = create_sequences(data_array, SEQ_LEN)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 8: TRAIN-TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

# -------------------------------
# STEP 9–12: TRAIN OR LOAD MODEL
# -------------------------------

if TRAIN_MODEL or not os.path.exists("aqi_model.h5"):
    print("\nTraining model...")

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(12, 4)))
    model.add(LSTM(32))
    model.add(Dense(3))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    model.save("aqi_model.h5")
    joblib.dump(scaler, "scaler.save")

    print("\nModel trained and saved!")

else:
    print("\nLoading existing model...")
    model = load_model("aqi_model.h5", compile=False)
    scaler = joblib.load("scaler.save")

# -------------------------------
# STEP 12: TEST PREDICTION
# -------------------------------

sample_input = X_test[-1]

pred = model.predict(sample_input.reshape(1, 12, 4))[0]

print("\nPredicted (scaled):", pred)

dummy = [0,0,0,0]
dummy[0:3] = pred

inv = scaler.inverse_transform([dummy])[0]

pred_pm25 = inv[0]
pred_pm10 = inv[1]
pred_temp = inv[2]

print("\nPredicted Values:")
print("PM2.5:", pred_pm25)
print("PM10:", pred_pm10)
print("Temperature:", pred_temp)

pred_aqi = calculate_aqi(pred_pm25, pred_pm10)

print("Predicted AQI:", pred_aqi)