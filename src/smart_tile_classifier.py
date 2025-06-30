import joblib
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tile_classifiers.presence_model import classify_presence, preprocess_tile_for_presence
from tile_classifiers.figure_color_model import classify_figure_color
from tile_classifiers.square_color_model import extract_features

SQUARE_COLOR_MODEL = joblib.load("models/square_color_model.pkl")
PRESENCE_MODEL = load_model("models/presence_model.h5")
FIGURE_COLOR_MODEL = load_model("models/figure_color_model.h5")

def classify_square_color(tile):
    feats = np.array(extract_features(tile)).reshape(1, -1)
    label = SQUARE_COLOR_MODEL.predict(feats)[0]
    return "white" if label == 1 else "black"

def classify_tile_full(tile):
    square_color = classify_square_color(tile)
    presence_input = preprocess_tile_for_presence(tile)
    presence_pred = PRESENCE_MODEL.predict(presence_input, verbose=0)[0][0]
    has_piece = presence_pred > 0.5

    if not has_piece:
        return f"{square_color}_empty"

    figure_color = classify_figure_color(tile)
    return f"{figure_color}_piece_on_{square_color}"

def classify_board(tiles, coords=None):
    results = {}
    if coords is None:
        coords = [f"{chr(ord('a') + col)}{8 - row}" for row in range(8) for col in range(8)]
    for coord, tile in zip(coords, tiles):
        results[coord] = classify_tile_full(tile)
    return results
