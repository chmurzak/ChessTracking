import cv2
import json
import os
import sys

# --- Konfiguracja ---
IMAGE_NAME = "0.jpg"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "chessboards", "data"))
IMAGE_PATH = os.path.join(DATA_DIR, IMAGE_NAME)
ANNOTATION_PATH = os.path.join(DATA_DIR, "corners_json", f"{os.path.splitext(IMAGE_NAME)[0]}_corners.json")
EXPECTED_POINTS = 4
POINT_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 255, 0)
POINT_RADIUS = 7
# ---------------------

print(f"✅ Sprawdzanie: {IMAGE_PATH}")
print(f"📄 JSON narożników: {ANNOTATION_PATH}")

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Błąd: obraz {IMAGE_PATH} nie istnieje."); sys.exit(1)
if not os.path.exists(ANNOTATION_PATH):
    print(f"❌ Błąd: adnotacja {ANNOTATION_PATH} nie istnieje."); sys.exit(1)

# Wczytaj obraz
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ Nie można wczytać obrazu: {IMAGE_PATH}")
    sys.exit(1)
h, w = image.shape[:2]

# Wczytaj JSON z narożnikami
with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    corners_data = json.load(f)

filename_from_json = corners_data.get("filename", "N/A")
print(f"🔎 Narożniki dla obrazu: {filename_from_json}")
found_corners = []

for corner_name, coords in corners_data.items():
    if corner_name == "filename":
        continue
    if not isinstance(coords, list) or len(coords) != 2:
        print(f"⚠️  Ignoruję niepoprawne dane narożnika: {corner_name} -> {coords}")
        continue

    x, y = coords
    try:
        x = int(x)
        y = int(y)
    except ValueError:
        print(f"⚠️  Nieprawidłowe współrzędne narożnika {corner_name}: {coords}")
        continue

    found_corners.append((corner_name, x, y))

# Wyświetl obraz z zaznaczonymi punktami
if not found_corners:
    print("❌ Nie znaleziono żadnych poprawnych narożników.")
    sys.exit(1)

print(f"✅ Znaleziono {len(found_corners)} narożniki:")
display = image.copy()
for name, x, y in found_corners:
    print(f"  {name}: ({x}, {y})")
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(display, (x, y), POINT_RADIUS, POINT_COLOR, -1)
        cv2.putText(display, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    else:
        print(f"⚠️  Punkt {name} poza zakresem ({x},{y}) względem ({w}x{h})")

# Pokaż obraz
cv2.imshow(f"🔍 Narożniki {IMAGE_NAME}", display)
print("Naciśnij Q, aby zamknąć okno...")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
