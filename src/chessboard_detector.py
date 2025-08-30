from ultralytics import YOLO
import numpy as np
import cv2
import os

model = YOLO("models/corner/best_corner_model.pt")

def complete_rectangle(points):
    if len(points) != 3:
        return points
    a, b, c = points
    dists = [
        np.linalg.norm(a - b) + np.linalg.norm(a - c),
        np.linalg.norm(b - a) + np.linalg.norm(b - c),
        np.linalg.norm(c - a) + np.linalg.norm(c - b)
    ]
    idx = np.argmin(dists)
    shared = points[idx]
    others = np.delete(points, idx, axis=0)
    p4 = others[0] + others[1] - shared
    return np.vstack([points, p4])

def detect_chessboard_corners(img: np.ndarray):
    results = model.predict(source=img, conf=0.3, save=False)[0]
    corners = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        corners.append([cx, cy])

    corners = np.array(corners)

    if len(corners) == 3:
        corners = complete_rectangle(corners)

    if len(corners) != 4:
        return False, corners

    return True, corners
