import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class SquareClassifier:


    def __init__(self, model_path="models/chess_piece_classifier.h5"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Brak pliku modelu: {model_path}")

        self.model = load_model(model_path)
        self.class_map = {
            0: "empty",
            1: "wP",
            2: "wR",
            3: "wN",
            4: "wB",
            5: "wQ",
            6: "wK",
            7: "bP",
            8: "bR",
            9: "bN",
            10: "bB",
            11: "bQ",
            12: "bK"
        }

    def classify_square(self, square_img):

        img_rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (50, 50))
        x = img_to_array(img_resized) / 255.0
        x = np.expand_dims(x, axis=0)  

        preds = self.model.predict(x)
        class_idx = np.argmax(preds[0])
        class_label = self.class_map.get(class_idx, "unknown")
        return class_label

    def classify_all_squares(self, squares):

        if len(squares) != 64:
            raise ValueError("Oczekiwano 64 fragmentów pól, otrzymano: {}".format(len(squares)))

        result = []
        idx = 0
        for row in range(8):
            row_labels = []
            for col in range(8):
                square_img = squares[idx]
                idx += 1
                label = self.classify_square(square_img)
                row_labels.append(label)
            result.append(row_labels)
        return result
