# import os
# import shutil
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tqdm import tqdm

# ORIGINAL_DIR = "dataset/pieces"
# TEMP_DIR = "dataset/_train_tmp"
# MODEL_PATH = "models/simple_CNN_for_class.h5"

# CLASSES = [
#     "white_piece_on_white", "white_piece_on_black",
#     "black_piece_on_white", "black_piece_on_black",
#     "white_empty", "black_empty"
# ]

# def map_to_class(square_color, piece_name):
#     if piece_name == "empty":
#         return f"{square_color}_empty"
#     elif "white" in piece_name:
#         return f"white_piece_on_{square_color}"
#     elif "black" in piece_name:
#         return f"black_piece_on_{square_color}"
#     else:
#         return None

# def prepare_dataset():
#     if os.path.exists(TEMP_DIR):
#         shutil.rmtree(TEMP_DIR)
#     os.makedirs(TEMP_DIR)

#     for square_color in ["white_squares", "black_squares"]:
#         square_dir = os.path.join(ORIGINAL_DIR, square_color)
#         for piece_dir in os.listdir(square_dir):
#             full_path = os.path.join(square_dir, piece_dir)
#             label = map_to_class(square_color.split("_")[0], piece_dir)
#             if label is None:
#                 continue
#             target_dir = os.path.join(TEMP_DIR, label)
#             os.makedirs(target_dir, exist_ok=True)
#             for img_file in os.listdir(full_path):
#                 src = os.path.join(full_path, img_file)
#                 dst = os.path.join(target_dir, f"{square_color}_{piece_dir}_{img_file}")
#                 shutil.copy2(src, dst)

# IMG_SIZE = (64, 64)
# BATCH_SIZE = 32
# EPOCHS = 12

# def train():
#     prepare_dataset()

#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         validation_split=0.2,
#         rotation_range=10,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         brightness_range=[0.8, 1.2],
#         zoom_range=0.1
#     )

#     train_data = datagen.flow_from_directory(
#         TEMP_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training'
#     )

#     val_data = datagen.flow_from_directory(
#         TEMP_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )

#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(train_data.num_classes, activation='softmax')
#     ])

#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
#     model.save(MODEL_PATH)
#     print(f"Zapisano model do: {MODEL_PATH}")

# if __name__ == "__main__":
#     train()
