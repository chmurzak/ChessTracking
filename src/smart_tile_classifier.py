import joblib
import numpy as np

from tile_classifiers.square_color_model import extract_features
from tile_classifiers.presence_model import (
    model as PRESENCE_MODEL,
    preprocess_tile_for_presence,
)
from tile_classifiers.figure_color_model import classify_figure_color

SQUARE_COLOR_MODEL = joblib.load("models/square_color_model.pkl")


def classify_square_color(tile):
    feats = np.array(extract_features(tile)).reshape(1, -1)
    label = SQUARE_COLOR_MODEL.predict(feats)[0]
    return "white" if label == 1 else "black"


def classify_tile_full(tile):

    square_color = classify_square_color(tile)

    presence_pred = PRESENCE_MODEL.predict(
        preprocess_tile_for_presence(tile), verbose=0
    )[0][0]
    if presence_pred <= 0.6:
        return f"{square_color}_empty"

    figure_color = classify_figure_color(tile)
    return f"{figure_color}_piece_on_{square_color}"


def classify_board(tiles, coords=None):

    if coords is None:
        coords = [f"{chr(ord('a') + col)}{8 - row}"
                  for row in range(8) for col in range(8)]

    assert len(tiles) == len(coords) == 64, "Lista kafelków i współrzędnych musi mieć 64 elementy"

    return {c: classify_tile_full(t) for c, t in zip(coords, tiles)}