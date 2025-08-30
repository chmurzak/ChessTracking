import os, time, cv2
from chessboard_detector import detect_chessboard_corners
from segmentation           import warp_and_segment_chessboard
from smart_tile_classifier  import classify_board
from orientation            import detect_orientation_direct, map_tiles_using_corners
from move_detection         import BoardState, board_dict_to_8x8, detect_simple_move
from move_tracker           import detect_move_full, apply_move_to_full

from utils.grid_calibration  import save_calibration, load_calibration
from utils.game_logic        import (
    save_game_state, load_start_position, fen_to_simple,
    model_label_to_simple, save_initial_states,
)
from utils.vision_helpers    import square_color, classify_promotion   
from pgn_writer              import update_pgn

DUMP_ROOT            = "dataset_auto/pieces"
START_POSITION_PATH  = "start_position.json"
GAME_STATES_DIR      = "debug/game_states"
CAM_URL              = "http://192.168.8.102:8080/video"


def main() -> None:
    print("ChessVision Start")

    cap = cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():
        print(f"Nie udało się otworzyć strumienia: {CAM_URL}")
        return
    print(f"Połączono z kamerą IP: {CAM_URL}")

    calibration_done = os.path.exists("debug/grid_calibration.json")
    calibration_corners = calibration_mapping = None
    if calibration_done:
        calibration_corners, calibration_mapping = load_calibration()

    board_state   = BoardState()
    os.makedirs("debug", exist_ok=True)
    save_initial_states(START_POSITION_PATH, GAME_STATES_DIR)

    start_pos   = load_start_position(START_POSITION_PATH)
    prev_simple = fen_to_simple(start_pos).copy()
    prev_full   = start_pos.copy()
    board_state.update_state(board_dict_to_8x8(start_pos))

    move_number = 0
    moves_path  = "debug/moves.txt"
    if os.path.exists(moves_path):
        os.remove(moves_path)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Brak klatek z kamery IP."); break

        cv2.imshow("Original Stream", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("k"):
            print("Tryb KALIBRACJI…")
            success, pts = detect_chessboard_corners(frame)
            if not success:
                print("Nie wykryto 4 narożników."); continue

            tiles_dict, _ = warp_and_segment_chessboard(frame, pts)
            tiles_list    = [tiles_dict[i] for i in range(64)]

            _ = classify_board(tiles_list) 
            corner_map = detect_orientation_direct(tiles_list)
            if corner_map is None: continue

            calibration_mapping = map_tiles_using_corners(corner_map)
            calibration_corners = pts; calibration_done = True
            save_calibration(pts, calibration_mapping)
            print("Kalibracja zapisana!")

        elif key == ord("d"):
            if not calibration_done:
                print("Najpierw kalibracja (K)!"); continue
            print("Segmentacja i klasyfikacja…")

            tiles_dict, warped = warp_and_segment_chessboard(
                frame, calibration_corners, tile_mapping=calibration_mapping
            )
            coords     = sorted(tiles_dict, key=lambda c: (int(c[1]), c[0]))
            tiles_list = [tiles_dict[c] for c in coords]

            raw_labels = classify_board(tiles_list, coords=coords)
            curr_simple = {
                c: model_label_to_simple(raw_labels.get(c, "empty"))
                for c in coords
            }

            for coord, tile in zip(coords, tiles_list):
                lab_piece = curr_simple[coord]
                col_sq    = square_color(coord)
                out_dir   = os.path.join(DUMP_ROOT, col_sq, lab_piece)
                os.makedirs(out_dir, exist_ok=True)
                fname = f"{coord}_{time.time_ns()}.png"
                cv2.imwrite(os.path.join(out_dir, fname), tile)


            curr_8x8  = board_dict_to_8x8(curr_simple)
            move_number += 1
            move_uci   = board_state.update_state(curr_8x8)
            if move_uci:
                print("Ruch:", move_uci)
                with open(moves_path, "a", encoding="utf-8") as f:
                    f.write(f"{move_number}: {move_uci}\n")

            save_game_state(curr_simple, move_number, "simple", GAME_STATES_DIR)

            move_full, piece_code = detect_move_full(prev_simple, curr_simple, prev_full)
            if move_full:
                frm_sq = move_full[:2]

                if prev_full[frm_sq].lower().endswith("p") and move_full[3] in ("1", "8"):
                    to_sq  = move_full[2:4]
                    tile   = tiles_dict[to_sq]
                    label, conf = classify_promotion(tile)

                    suf = {"queen":"q", "rook":"r", "knight":"n", "bishop":"b"}[label]
                    if prev_full[frm_sq].isupper():        
                        suf = suf.upper()

                    if len(move_full) == 5 and move_full[-1].lower() in ("q","r","n","b"):
                        move_full = move_full[:4]

                    move_full += suf.lower()               
                    piece_code = suf

                    print(f" PROMOCJA: {label.upper():7} ({conf*100:5.1f}% pewności)")

                print(f"(Full) ruch: {move_full} ({piece_code})")
                apply_move_to_full(prev_full, move_full, piece_code)


                with open(moves_path, "a", encoding="utf-8") as f:
                  f.write(f"{move_number}: {move_full}\n")                

            move_simple, piece = detect_simple_move(prev_simple, curr_simple)
            if move_simple:
                with open(moves_path, "a", encoding="utf-8") as f:
                    f.write(f"{move_number}: {move_simple} ({piece})\n")

            update_pgn()
            prev_simple = curr_simple.copy()

            cv2.imshow("Warped Chessboard", warped)
            cv2.imwrite("debug/warped_board.png", warped)

        elif key == ord("q"):
            print("Zamykanie ChessVision…"); break

    cap.release(); cv2.destroyAllWindows()
    print("ChessVision zakończone.")


if __name__ == "__main__":
    main()
