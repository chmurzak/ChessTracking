import cv2
import json
import numpy as np
import os

IMG_NAME = "0.jpg"
JSON_NAME = "0_corners.json"

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "chessboards", "data"))
IMG_PATH = os.path.join(DATA_DIR, IMG_NAME)
JSON_PATH = os.path.join(DATA_DIR, "corners_json", JSON_NAME)

BOARD_SIZE = 8
COLOR_LINE = (0, 255, 0)
COLOR_TEXT = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
RADIUS = 2

img = cv2.imread(IMG_PATH)
if img is None:
    print("Nie znaleziono obrazu.")
    exit()

h, w = img.shape[:2]

with open(JSON_PATH, "r") as f:
    corners = json.load(f)

try:
    A1 = np.array(corners["A1"], dtype=np.float32)
    A8 = np.array(corners["A8"], dtype=np.float32)
    H1 = np.array(corners["H1"], dtype=np.float32)
    H8 = np.array(corners["H8"], dtype=np.float32)
except KeyError as e:
    print(f"Brak narożnika w pliku JSON: {e}")
    exit()


grid = []

for row in range(BOARD_SIZE + 1):
    alpha = row / BOARD_SIZE
    left = (1 - alpha) * A8 + alpha * A1
    right = (1 - alpha) * H8 + alpha * H1
    row_points = []
    for col in range(BOARD_SIZE + 1):
        beta = col / BOARD_SIZE
        pt = (1 - beta) * left + beta * right
        row_points.append(pt)
    grid.append(row_points)

img_grid = img.copy()
for row in range(BOARD_SIZE + 1):
    for col in range(BOARD_SIZE + 1):
        pt = tuple(np.round(grid[row][col]).astype(int))
        cv2.circle(img_grid, pt, RADIUS, COLOR_LINE, -1)

for row in range(BOARD_SIZE + 1):
    for col in range(BOARD_SIZE):
        pt1 = tuple(np.round(grid[row][col]).astype(int))
        pt2 = tuple(np.round(grid[row][col + 1]).astype(int))
        cv2.line(img_grid, pt1, pt2, COLOR_LINE, 1)

for col in range(BOARD_SIZE + 1):
    for row in range(BOARD_SIZE):
        pt1 = tuple(np.round(grid[row][col]).astype(int))
        pt2 = tuple(np.round(grid[row + 1][col]).astype(int))
        cv2.line(img_grid, pt1, pt2, COLOR_LINE, 1)

letters = "ABCDEFGH"
for row in range(BOARD_SIZE):
    for col in range(BOARD_SIZE):
        pt1 = grid[row][col]
        pt2 = grid[row + 1][col + 1]
        center = ((pt1 + pt2) / 2).astype(int)
        label = f"{letters[col]}{BOARD_SIZE - row}"
        cv2.putText(img_grid, label, tuple(center), FONT, 0.5, COLOR_TEXT, 1)

cv2.imshow("🟩 Segmentacja szachownicy", img_grid)
print("✅ Naciśnij Q, aby zamknąć")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
