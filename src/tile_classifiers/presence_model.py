import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/presence_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_tile_for_presence(tile):
    """Zamienia obraz pola na przetworzoną wersję z konturami (krawędzie Canny)."""
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.merge([edges] * 3)  # przekształcenie do 3 kanałów
    resized = cv2.resize(edges_rgb, (64, 64))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)

def classify_presence(tile):
    """Zwraca 'occupied' jeśli figura jest na polu, inaczej 'empty'."""
    input_data = preprocess_tile_for_presence(tile)
    prediction = model.predict(input_data, verbose=0)[0][0]
    return "occupied" if prediction > 0.5 else "empty"
