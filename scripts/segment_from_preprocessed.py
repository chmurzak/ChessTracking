import os
import cv2
import numpy as np
from pathlib import Path

input_dir = Path("dataset/labeled_preprocessed")
output_base = Path("dataset/pieces")
output_base.mkdir(parents=True, exist_ok=True)

tile_size = 64
board_size = tile_size * 8  

def segment_image_to_tiles(image_path, output_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Nie można wczytać obrazu: {image_path}")
        return
    image = cv2.resize(image, (board_size, board_size))

    os.makedirs(output_dir, exist_ok=True)
    for row in range(8):
        for col in range(8):
            y1, y2 = row * tile_size, (row + 1) * tile_size
            x1, x2 = col * tile_size, (col + 1) * tile_size
            tile = image[y1:y2, x1:x2]
            tile_name = f"{chr(ord('A') + col)}{8 - row}.png"
            tile_path = output_dir / tile_name
            cv2.imwrite(str(tile_path), tile)

def main():
    all_images = sorted(input_dir.glob("*.png"))
    print(f"Znaleziono {len(all_images)} obrazów do segmentacji...")

    for i, image_path in enumerate(all_images, 1):
        name = image_path.stem
        out_dir = output_base / name
        segment_image_to_tiles(image_path, out_dir)
        print(f"[{i}/{len(all_images)}] ✅ {image_path.name} → {out_dir}")

if __name__ == "__main__":
    main()
