import numpy as np
from smart_tile_classifier import classify_tile_full

_LABEL2CANON = {
    "white_piece_on_black": "A1",
    "white_piece_on_white": "H1",
    "black_piece_on_white": "A8",
    "black_piece_on_black": "H8",
}
_CORNER_TILES = [0, 7, 56, 63]


def detect_orientation_direct(tiles: list):
    """
    Zwraca dict {kanoniczny_róg: index_tile}.
    """
    canon_to_tile = {}
    print("== Klasyfikacja narożnych kafelków ==")
    for idx in _CORNER_TILES:
        label = classify_tile_full(tiles[idx])
        print(f"  tile_{idx:02}: {label}")
        if label in _LABEL2CANON:
            canon_to_tile[_LABEL2CANON[label]] = idx

    if len(canon_to_tile) < 4:
        print("Nie udało się jednoznacznie zidentyfikować wszystkich narożników.")
        return None

    print("Zidentyfikowane narożniki:")
    for canon, idx in canon_to_tile.items():
        print(f"  {canon} ← tile_{idx:02}")
    return canon_to_tile


def map_tiles_using_corners(corner_map: dict[str, int]) -> dict[int, str]:
    """
    Buduje mapping index 'a1'...'h8' dla **dowolnego obrotu 0/90/180/270 °**.
    """
    a1 = corner_map["A1"]
    h1 = corner_map["H1"]
    a8 = corner_map["A8"]

    ax, ay = a1 % 8, a1 // 8
    hx, hy = h1 % 8, h1 // 8
    a8x, a8y = a8 % 8, a8 // 8

    vx = np.sign(hx - ax), np.sign(hy - ay)     
    vy = np.sign(a8x - ax), np.sign(a8y - ay)    

    files = "abcdefgh"
    ranks = "12345678"

    mapping: dict[int, str] = {}
    for r in range(8):            
        for f in range(8):        
            x = ax + vx[0] * f + vy[0] * r
            y = ay + vx[1] * f + vy[1] * r
            idx = y * 8 + x
            mapping[idx] = f"{files[f]}{ranks[r]}"
    return mapping
