import cv2, numpy as np
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

MODEL_PATH = "models/piece_color/effnet_b0/model.h5"

model = load_model(MODEL_PATH, compile=False)   

IMG_SIZE = 64
THRESH   = 0.90      # Najlepsze F1

def preprocess_tile_for_figure_color(tile: np.ndarray) -> np.ndarray:
    rgb  = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")
    rgb  = preprocess_input(rgb)                
    return np.expand_dims(rgb, axis=0)          

def classify_figure_color(tile: np.ndarray) -> str:
    prob = model.predict(preprocess_tile_for_figure_color(tile), verbose=0)[0, 0]
    return "white" if prob > THRESH else "black"