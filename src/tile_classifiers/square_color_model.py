import cv2
import joblib
import numpy as np

MODEL_PATH = "models/square_color_model.pkl"
model = joblib.load(MODEL_PATH)

def extract_features(img):
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img is None or len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"extract_features: input image has wrong shape: {None if img is None else img.shape}")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    l_mean, l_std = lab[..., 0].mean(), lab[..., 0].std()
    h_mean        = hsv[..., 0].mean()
    s_mean, s_std = hsv[..., 1].mean(), hsv[..., 1].std()
    v_mean, v_std = hsv[..., 2].mean(), hsv[..., 2].std()
    dark_ratio    = np.sum(gray < 80) / gray.size

    return [l_mean, l_std, h_mean, s_mean, s_std, v_mean, v_std, dark_ratio]


def classify_square_color(tile):
    feats = np.array(extract_features(tile)).reshape(1, -1)
    label = model.predict(feats)[0]
    return "white" if label == 1 else "black"
