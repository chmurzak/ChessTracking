import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolo_chess.pt')

def detect_pieces_yolo(frame):
    """ Wykrywa figury szachowe na podstawie modelu YOLOv5 """
    results = model(frame)
    detected_pieces = []

    for *xyxy, conf, cls in results.xyxy[0]:  # Pobranie danych detekcji
        x1, y1, x2, y2 = map(int, xyxy)
        piece = model.names[int(cls)]  # Pobranie nazwy figury
        detected_pieces.append((piece, (x1, y1, x2, y2)))

        # Rysowanie bounding boxów na obrazie
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, piece, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_pieces
