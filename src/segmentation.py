import cv2
import numpy as np
import os

def order_corners(corners):
    corners = np.array(corners, dtype="float32")
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = corners[np.argmin(s)]       
    ordered[2] = corners[np.argmax(s)]       
    ordered[1] = corners[np.argmin(diff)]    
    ordered[3] = corners[np.argmax(diff)]    
    return ordered

def warp_and_segment_chessboard(image, corners, output_dir="debug/tiles", size=512, tile_mapping=None):
    if len(corners) != 4:
        raise ValueError("Potrzeba dokładnie 4 narożników do przekształcenia")

    ordered = order_corners(corners)
    dst_pts = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(image, M, (size, size))

    os.makedirs(output_dir, exist_ok=True)
    tile_size = size // 8

    tiles = {}
    for row in range(8):
        for col in range(8):
            x1, y1 = col * tile_size, row * tile_size
            x2, y2 = x1 + tile_size, y1 + tile_size
            tile = warped[y1:y2, x1:x2]
            index = row * 8 + col

            key = tile_mapping[index] if tile_mapping and index in tile_mapping else index

            tiles[key] = tile
            tile_path = os.path.join(output_dir, f"{key}.png")
            cv2.imwrite(tile_path, tile)

    return tiles, warped
