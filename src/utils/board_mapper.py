import os
import cv2
import json

def save_debug_board(tiles, labels, mapping, output_dir="debug/chess_squares"):
    
    os.makedirs(output_dir, exist_ok=True)

    image_map, label_map = {}, {}

    for idx, coord in mapping.items():
        label = labels.get(idx, "empty")      
        cv2.imwrite(os.path.join(output_dir, f"{coord}_{label}.png"), tiles[idx])

        image_map[coord] = idx
        label_map[coord] = label

    with open(os.path.join(output_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"tile_index_mapping": image_map, "labels": label_map},
            f, indent=2, ensure_ascii=False
        )

def save_tiles_by_coords(tiles, tile_mapping, output_dir="debug/chess_squares"):

    os.makedirs(output_dir, exist_ok=True)
    for idx, coord in tile_mapping.items():
        tile = tiles[idx]
        cv2.imwrite(os.path.join(output_dir, f"{coord}.png"), tile)

def get_sorted_squares(squares):

    def chess_sort_key(sq):
        return (8 - int(sq[1]), ord(sq[0]) - ord('a'))
    return sorted(squares, key=chess_sort_key)
