import os
import sys
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chessboard_detector import detect_chessboard_corners
from src.segmentation import warp_and_segment_chessboard


INPUT_DIR = Path("dataset/corners/raw_data/chessboards/data")
OUTPUT_BASE = Path("dataset/pieces")

STEP = 10

def main():
    images = sorted(INPUT_DIR.glob("*.jpg"))[::STEP]
    print(f"Znaleziono {len(images)} zdjęć do przetworzenia (co {STEP})")

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Przetwarzanie {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Błąd wczytywania obrazu: {img_path}")
            continue

        success, corners = detect_chessboard_corners(img)
        if not success:
            print(f"Nie wykryto 4 narożników w {img_path.name}")
            continue

        out_dir = OUTPUT_BASE / img_path.stem
        warp_and_segment_chessboard(img, corners, output_dir=str(out_dir))

if __name__ == "__main__":
    main()
