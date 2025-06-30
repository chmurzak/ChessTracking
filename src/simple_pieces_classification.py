# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import cv2

# MODEL_PATH = "models/simple_CNN_for_class.h5"
# LABELS = [
#     "white_piece_on_white", "white_piece_on_black",
#     "black_piece_on_white", "black_piece_on_black",
#     "white_empty", "black_empty"
# ]

# def load_piece_model():
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
#     return load_model(MODEL_PATH)

# def preprocess_tile(tile, target_size=(64, 64)):
#     image = cv2.resize(tile, target_size)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = img_to_array(image) / 255.0  # Normalizacja
#     return np.expand_dims(image, axis=0)  # [1, 64, 64, 3]

# def classify_tile(tile, model):
#     input_data = preprocess_tile(tile)
#     prediction = model.predict(input_data, verbose=0)[0]
#     predicted_class = LABELS[np.argmax(prediction)]
#     return predicted_class

# def classify_all_tiles(tiles, model):
#     return [classify_tile(tile, model) for tile in tiles]

# # Mapowanie 64 pól do notacji szachowej
# def generate_board_mapping():
#     return [f"{chr(ord('a') + col)}{8 - row}" for row in range(8) for col in range(8)]

# # Zwraca słownik: {"a2": "white_piece_on_black", ..., "e4": "black_empty"}
# def get_classified_board(tiles, model):
#     labels = classify_all_tiles(tiles, model)
#     mapping = generate_board_mapping()
#     return dict(zip(mapping, labels))
