import os
import cv2

from chessboard_detector import detect_chessboard_corners
from segmentation import warp_and_segment_chessboard
from smart_tile_classifier import classify_board
from orientation import detect_orientation_direct, map_tiles_using_corners

from utils.board_mapper       import save_tiles_by_coords
from utils.grid_calibration   import save_calibration, load_calibration
from utils.game_logic         import (
    save_game_state, load_start_position, fen_to_simple,
    model_label_to_simple, save_initial_states,
)
from move_detection           import (
    BoardState, board_dict_to_8x8, detect_simple_move,
)

START_POSITION_PATH = "start_position.json"
GAME_STATES_DIR     = "debug/game_states"

def main() -> None:
    print("ChessVision Start")
    CAM_URL = "http://192.168.8.102:8080/video"

    cap = cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():
        print(f"Nie udało się otworzyć strumienia: {CAM_URL}")
        return
    print(f"Połączono z kamerą IP: {CAM_URL}")

    calibration_done      = os.path.exists("debug/grid_calibration.json")
    calibration_corners   = None
    calibration_mapping   = None

    if calibration_done:
        calibration_corners, calibration_mapping = load_calibration()

    board_state = BoardState()
    os.makedirs("debug", exist_ok=True)
    save_initial_states(START_POSITION_PATH, GAME_STATES_DIR)

    start_pos  = load_start_position(START_POSITION_PATH)
    board_state.update_state(board_dict_to_8x8(start_pos))
    prev_simple = fen_to_simple(start_pos).copy()
    move_number = 0
    moves_path  = "debug/moves.txt"
    if os.path.exists(moves_path):
        os.remove(moves_path)

    # ─── pętla główna ─────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Brak klatek z kamery IP.")
            break

        cv2.imshow("Original Stream", frame)
        key = cv2.waitKey(1) & 0xFF


        if key == ord("k"):
            print("Tryb KALIBRACJI…")
            success, pts = detect_chessboard_corners(frame)
            if not success:
                print("Nie wykryto 4 narożników.")
                continue

            tiles_dict, warped = warp_and_segment_chessboard(frame, pts,
                                                             output_dir="debug/tiles")
            tiles_list = [tiles_dict[i] for i in range(64)]

            _ = classify_board(tiles_list)             
            corner_map = detect_orientation_direct(tiles_list)
            if corner_map is None:
                continue

            calibration_mapping = map_tiles_using_corners(corner_map)
            calibration_corners = pts
            calibration_done    = True

            save_calibration(pts, calibration_mapping)
            print("Kalibracja zapisana!")

            save_tiles_by_coords(tiles_list, calibration_mapping,
                                 output_dir="debug/chess_squares")
            cv2.imwrite("debug/warped_board.png", warped)


        elif key == ord("d"):
            if not calibration_done:
                print("Najpierw wykonaj kalibrację (klawisz K)!")
                continue
            print("Segmentacja i klasyfikacja…")

            tiles_dict, warped = warp_and_segment_chessboard(
                frame, calibration_corners, output_dir="debug/tiles",
            )
            tiles_list = [tiles_dict[i] for i in range(64)]

            save_tiles_by_coords(tiles_list, calibration_mapping,
                                 output_dir="debug/chess_squares")

            coords     = [calibration_mapping[i] for i in range(64)]
            raw_labels = classify_board(tiles_list, coords=coords)

            curr_simple = {
                coord: model_label_to_simple(raw_labels.get(coord, "empty"))
                for coord in coords
            }
            print("curr_simple:", curr_simple)

            curr_8x8 = board_dict_to_8x8(curr_simple)
            move = board_state.update_state(curr_8x8)
            move_number += 1
            if move:
                print("Wykryty ruch:", move)
                with open(moves_path, "a", encoding="utf-8") as f:
                    f.write(f"{move_number}: {move}\n")

            save_game_state(curr_simple, move_number, "simple",
                            dir_path=GAME_STATES_DIR)

            move_simple, piece = detect_simple_move(prev_simple, curr_simple)
            if move_simple:
                print(f"(Prosty) ruch: {move_simple} ({piece})")
                with open(moves_path, "a", encoding="utf-8") as f:
                    f.write(f"{move_number}: {move_simple} ({piece})\n")
            prev_simple = curr_simple.copy()

            cv2.imshow("Warped Chessboard", warped)
            cv2.imwrite("debug/warped_board.png", warped)

        elif key == ord("q"):
            print("Zamykanie ChessVision…")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("ChessVision zakończone.")


if __name__ == "__main__":
    main()
