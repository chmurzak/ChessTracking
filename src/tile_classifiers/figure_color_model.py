import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/figure_color_model.h5"
model = load_model(MODEL_PATH)

def preprocess_tile_for_figure_color(tile):
    resized = cv2.resize(tile, (64, 64))
    image = resized.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def classify_figure_color(tile):
    """Zwraca 'white' lub 'black' kolor figury, jeśli jest obecna."""
    input_data = preprocess_tile_for_figure_color(tile)
    prediction = model.predict(input_data, verbose=0)[0][0]
    return "white" if prediction > 0.5 else "black"
