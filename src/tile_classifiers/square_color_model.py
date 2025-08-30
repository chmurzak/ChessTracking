import cv2
import joblib
import numpy as np

MODEL_PATH = "models/square_color/model.pkl"
model = joblib.load(MODEL_PATH)

def extract_features(img):
    
    if img is None:
        raise ValueError("extract_features: input image is None")
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    L = lab[..., 0].astype(np.float32)
    H = hsv[..., 0].astype(np.float32)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)

    bright_ratio = float((V > 200).sum()) / V.size
    p25, p75     = np.percentile(L, [25, 75])
    L_IQR        = float(p75 - p25)

    return [
        float(L.mean()), float(L.std()),
        float(H.mean()),
        float(S.mean()), float(S.std()),
        float(V.mean()), float(V.std()),
        bright_ratio, L_IQR
    ]

def classify_square_color(tile):
    feats = np.array(extract_features(tile)).reshape(1, -1)
    label = model.predict(feats)[0]
    return "white" if label == 1 else "black"
