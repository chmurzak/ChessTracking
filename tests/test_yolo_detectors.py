from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("models/best.pt")
img = cv2.imread("temp_chess.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model.predict(source=img_rgb, conf=0.3, save=False)[0]

corners = []
for box in results.boxes.xyxy:
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    corners.append([cx, cy])

corners = np.array(corners)
print(f"🧩 Wykryto {len(corners)} narożników")

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

if len(corners) == 3:
    corners = complete_rectangle(corners)
    print("➕ Dodano brakujący narożnik przez dopełnienie prostokąta")

os.makedirs("debug", exist_ok=True)
for i, (x, y) in enumerate(corners):
    color = (0, 255, 255) if i >= 3 else (0, 255, 0)
    cv2.circle(img, (int(x), int(y)), 10, color, -1)

cv2.imwrite("debug/corners_detected.jpg", img)
print("✅ Zapisano debug/corners_detected.jpg")
