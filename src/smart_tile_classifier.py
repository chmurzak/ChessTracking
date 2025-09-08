import numpy as np
from utils.vision_helpers import square_color  

from tile_classifiers.square_color_model import (
    extract_features,
    model as SQUARE_COLOR_MODEL,     
    classify_square_color,            
)
from tile_classifiers.presence_model import (
    model as PRESENCE_MODEL,
    preprocess_tile_for_presence,
    PRESENCE_THRESH,
)
from tile_classifiers.figure_color_model import (
    model as FIGCOLOR_MODEL,
    preprocess_tile_for_figure_color,
    THRESH as FIGCOLOR_THRESH,
)


def classify_tile_full(tile):
    sq = classify_square_color(tile)
    pr = PRESENCE_MODEL.predict(preprocess_tile_for_presence(tile), verbose=0)[0][0]
    if pr <= PRESENCE_THRESH:
        return f"{sq}_empty"
    fc_prob = FIGCOLOR_MODEL.predict(
        preprocess_tile_for_figure_color(tile), batch_size=1, verbose=0
    )[0][0]
    fc = "white" if fc_prob > FIGCOLOR_THRESH else "black"
    return f"{fc}_piece_on_{sq}"

def classify_board(tiles, coords=None, force_sqcolor_model=False):

    assert len(tiles) == 64, "Musi być 64 tiles."

    if coords is not None and not force_sqcolor_model:
        sq_lbl = np.array([square_color(c) for c in coords])
    else:
        Xsq    = np.array([extract_features(t) for t in tiles], dtype=np.float32)
        sq_bin = SQUARE_COLOR_MODEL.predict(Xsq)
        sq_lbl = np.where(sq_bin == 1, "white", "black")

    Xpr     = np.concatenate([preprocess_tile_for_presence(t) for t in tiles], axis=0)
    pr_prob = PRESENCE_MODEL.predict(Xpr, batch_size=64, verbose=0).reshape(-1)
    present = pr_prob > PRESENCE_THRESH

    result = {}
    if present.any():
        idxs    = np.where(present)[0]
        Xfc     = np.concatenate([preprocess_tile_for_figure_color(tiles[i]) for i in idxs], axis=0)
        fc_prob = FIGCOLOR_MODEL.predict(Xfc, batch_size=64, verbose=0).reshape(-1)
        fc_lbls = np.where(fc_prob > FIGCOLOR_THRESH, "white", "black")
        for j, i in enumerate(idxs):
            key = coords[i] if coords is not None else i
            result[key] = f"{fc_lbls[j]}_piece_on_{sq_lbl[i]}"

    for i in range(64):
        if not present[i]:
            key = coords[i] if coords is not None else i
            result[key] = f"{sq_lbl[i]}_empty"

    return result
