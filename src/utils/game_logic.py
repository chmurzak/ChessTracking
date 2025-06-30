import os
import json

def save_game_state(state_dict, move_number, suffix="", dir_path="debug/game_states"):
    os.makedirs(dir_path, exist_ok=True)
    if suffix:
        out_path = os.path.join(dir_path, f"{move_number}_{suffix}.json")
    else:
        out_path = os.path.join(dir_path, f"{move_number}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)

def load_start_position(path="start_position.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fen_to_simple(board_dict):
    out = {}
    for sq, val in board_dict.items():
        if isinstance(val, str) and val.startswith("w"):
            out[sq] = "white_piece"
        elif isinstance(val, str) and val.startswith("b"):
            out[sq] = "black_piece"
        else:
            out[sq] = "empty"
    return out

def model_label_to_simple(label):
    if "white_piece" in label:
        return "white_piece"
    if "black_piece" in label:
        return "black_piece"
    return "empty"

def save_initial_states(start_position_path="start_position.json", game_states_dir="debug/game_states"):
    start_position = load_start_position(start_position_path)
    save_game_state(start_position, 0, "full", dir_path=game_states_dir)
    start_simple = fen_to_simple(start_position)
    save_game_state(start_simple, 0, "simple", dir_path=game_states_dir)
    print(f"Stany początkowe zapisane jako 0_full.json i 0_simple.json w {game_states_dir}")
