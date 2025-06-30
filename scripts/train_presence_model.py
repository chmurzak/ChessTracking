import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

MODEL_OUT = "models/presence_model.h5"

WHITE_PATH = "dataset/pieces/white_squares"
BLACK_PATH = "dataset/pieces/black_squares"

def load_presence_dataset():
    X, y = [], []

    for label, base in [(0, WHITE_PATH), (0, BLACK_PATH)]:
        for fname in os.listdir(os.path.join(base, "empty")):
            fpath = os.path.join(base, "empty", fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_rgb = cv2.merge([edges] * 3)
            edges_rgb = cv2.resize(edges_rgb, (64, 64))
            X.append(edges_rgb)
            y.append(0)

        for sub in os.listdir(base):
            if sub == "empty":
                continue
            folder = os.path.join(base, sub)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges_rgb = cv2.merge([edges] * 3)
                edges_rgb = cv2.resize(edges_rgb, (64, 64))
                X.append(edges_rgb)
                y.append(1)

    X = np.array(X).astype("float32") / 255.0
    y = np.array(y).astype("float32")
    return X, y


def train():
    X, y = load_presence_dataset()
    print(f"Załadowano: {len(X)} przykładów | zajęte: {(y==1).sum()} | puste: {(y==0).sum()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10,
              validation_data=(X_val, y_val),
              batch_size=32)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"Zapisano → {MODEL_OUT}")


if __name__ == "__main__":
    train()
