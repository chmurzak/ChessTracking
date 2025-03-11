import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolo_chess.pt')

def detect_pieces_yolo(frame):
    """ Wykrywa figury szachowe na podstawie modelu YOLOv5 """
    results = model(frame)
    detected_pieces = []

    for *xyxy, conf, cls in results.xyxy[0]:  # Pobranie danych detekcji
        x1, y1, x2, y2 = map(int, xyxy)
        piece = model.names[int(cls)]  # Pobranie nazwy figury
        detected_pieces.append({"id": None, "bbox": (x1, y1, x2, y2), "piece": piece})

    return frame, detected_pieces
