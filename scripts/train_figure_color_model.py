import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

WHITE_PATH = "dataset/pieces/white_squares"
BLACK_PATH = "dataset/pieces/black_squares"
MODEL_PATH = "models/figure_color_model.h5"


def load_figure_color_dataset():
    X, y = [], []

    for base_path in [WHITE_PATH, BLACK_PATH]:
        for sub in os.listdir(base_path):
            full_path = os.path.join(base_path, sub)
            if sub == "empty" or not os.path.isdir(full_path):
                continue
            label = 1 if sub.startswith("white_") else 0
            for fname in os.listdir(full_path):
                fpath = os.path.join(full_path, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
                X.append(img)
                y.append(label)

    return np.array(X), np.array(y)


def train():
    X, y = load_figure_color_dataset()
    print(f"Załadowano: {len(X)} przykładów | białe figury: {(y==1).sum()} | czarne figury: {(y==0).sum()}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val), batch_size=32)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model zapisany do: {MODEL_PATH}")


if __name__ == "__main__":
    train()
