import tensorflow as tf

# load your existing model
model = tf.keras.models.load_model("aqi_model.keras", compile=False)

# re-save in clean compatible format
model.save("aqi_model_fixed.keras")