# import cv2
# import numpy as np
# from tensorflow import keras

# MODEL_PATH = "models/model_b.h5"
# model = keras.models.load_model(MODEL_PATH)

# CLASS_NAMES = [
#     'empty', 'bishop_b', 'bishop_w', 'king_b', 'king_w', 'knight_b', 'knight_w',
#     'pawn_b', 'pawn_w', 'queen_b', 'queen_w', 'rook_b', 'rook_w'
# ]

# def preprocess_image(image, target_size=(512, 512)):
#     image = cv2.resize(image, target_size)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image / 255.0
#     return image

# def analyze_frame(image_path):
#     image = cv2.imread(image_path)
#     image = preprocess_image(image)

#     prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
#     prediction = prediction.reshape((8, 8, len(CLASS_NAMES)))

#     result = {}
#     for row in range(8):
#         for col in range(8):
#             class_id = np.argmax(prediction[row, col])
#             label = CLASS_NAMES[class_id]
#             coord = chr(ord('A') + col) + str(8 - row)
#             result[coord] = label

#     return result
