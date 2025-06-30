import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

WHITE_BASE = "dataset/pieces/white_squares"
BLACK_BASE = "dataset/pieces/black_squares"
MODEL_OUT  = "models/square_color_model.pkl"


def extract_features(img: np.ndarray) -> list[float]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    l_mean, l_std = lab[..., 0].mean(), lab[..., 0].std()
    h_mean        = hsv[..., 0].mean()
    s_mean, s_std = hsv[..., 1].mean(), hsv[..., 1].std()
    v_mean, v_std = hsv[..., 2].mean(), hsv[..., 2].std()
    dark_ratio    = np.sum(gray < 80) / gray.size

    return [l_mean, l_std, h_mean, s_mean, s_std, v_mean, v_std, dark_ratio]


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for label, base in [(0, BLACK_BASE), (1, WHITE_BASE)]:
        for subdir in os.listdir(base):
            full_path = os.path.join(base, subdir)
            if not os.path.isdir(full_path):
                continue
            for fname in os.listdir(full_path):
                fpath = os.path.join(full_path, fname)
                img = cv2.imread(fpath)
                if img is None or len(img.shape) != 3 or img.shape[2] != 3:
                    print(f"Błąd wczytania obrazu: {fpath}")
                    continue
                img = cv2.resize(img, (64, 64))
                X.append(extract_features(img))
                y.append(label)

    return np.array(X), np.array(y)


def train():
    X, y = load_dataset()
    print(f"Załadowano {len(X)} przykładów (czarne: {(y==0).sum()} | białe: {(y==1).sum()})")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    print("Walidacja (20% split):\n",
          classification_report(y_val, model.predict(X_val), target_names=["black", "white"]))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"Cross-val F1 (5-fold): {scores.mean():.4f} ± {scores.std():.4f}")

    feat_names = ['l_mean', 'l_std', 'h_mean', 's_mean', 's_std', 'v_mean', 'v_std', 'dark_ratio']
    importances = model.feature_importances_
    plt.barh(feat_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Cechy koloru pola")
    plt.tight_layout()
    plt.show()

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Model zapisany do: {MODEL_OUT}")


if __name__ == "__main__":
    train()
