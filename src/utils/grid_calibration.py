import json
import os

def save_calibration(corners, tile_mapping, path="debug/grid_calibration.json"):
    data = {
        "corners": [list(map(float, pt)) for pt in corners],  
        "tile_mapping": {str(k): v for k, v in tile_mapping.items()}  
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_calibration(path="debug/grid_calibration.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    corners = [tuple(pt) for pt in data["corners"]]
    tile_mapping = {int(k): v for k, v in data["tile_mapping"].items()}
    return corners, tile_mapping
