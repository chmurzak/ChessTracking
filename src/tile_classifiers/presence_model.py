import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/presence/presence_lightcnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (64, 64)
PRESENCE_THRESH = 0.8

def preprocess_tile_for_presence(tile: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
    lab = cv2.resize(lab, IMG_SIZE, interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    return np.expand_dims(lab, axis=0)

def classify_presence(tile: np.ndarray) -> str:
    prob = model.predict(preprocess_tile_for_presence(tile), verbose=0)[0, 0]
    return "occupied" if prob > PRESENCE_THRESH else "empty"