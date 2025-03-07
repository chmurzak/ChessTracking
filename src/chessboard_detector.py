import cv2
import numpy as np

def detect_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame
import cv2
import numpy as np
import tensorflow as tf
import chess.pgn
import chess
from utils import get_chessboard_corners, convert_position_to_algebraic

# Wczytanie modelu
model = tf.keras.models.load_model("models/chess_piece_classifier.h5")
CATEGORIES = ["pawn", "rook", "knight", "bishop", "queen", "king", "empty"]

# Przekształcenie perspektywy do rzutu prostopadłego
def apply_perspective_transform(frame, corners):
    if corners is None or len(corners) < 4:
        return None
    
    corners = corners.reshape(-1, 2)
    src_pts = np.float32([corners[0], corners[6], corners[-1], corners[-7]])
    dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, M, (300, 300))

# Podział na 64 pola
def split_chessboard_into_squares(warped):
    if warped is None:
        return None, []
    square_size = 300 // 8
    squares = []
    for row in range(8):
        for col in range(8):
            x_start, y_start = col * square_size, row * square_size
            x_end, y_end = x_start + square_size, y_start + square_size
            square = warped[y_start:y_end, x_start:x_end]
            squares.append((row, col, square))
            cv2.rectangle(warped, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
    return warped, squares

# Klasyfikacja figury za pomocą modelu CNN
def classify_piece(square):
    img = cv2.resize(square, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)
    prediction = model.predict(img)
    return CATEGORIES[np.argmax(prediction)]

# Wykrywanie figur na planszy
def detect_pieces(squares):
    piece_positions = {}
    for row, col, square in squares:
        piece = classify_piece(square)
        if piece != "empty":
            piece_positions[(row, col)] = piece[0].upper()
    return piece_positions

# Śledzenie ruchów figur
def track_piece_movements(previous_positions, current_positions, game_moves):
    movements = []
    for old_pos in previous_positions:
        if old_pos not in current_positions:
            for new_pos in current_positions:
                if new_pos not in previous_positions:
                    piece = previous_positions[old_pos]
                    old_alg = convert_position_to_algebraic(*old_pos)
                    new_alg = convert_position_to_algebraic(*new_pos)
                    move_str = f"{piece}{old_alg}{new_alg}"
                    game_moves.append(move_str)
                    movements.append((piece, old_alg, new_alg))
                    break
    return movements

# Zapisywanie partii do PGN
def save_pgn(game_moves, filename="game.pgn"):
    game = chess.pgn.Game()
    node = game
    for move in game_moves:
        node = node.add_variation(chess.Move.null())
        node.comment = move
    with open(filename, "w") as pgn_file:
        pgn_file.write(str(game))

# Główna pętla wykrywania

def detect_chessboard_from_camera():
    cap = cv2.VideoCapture(0)
    previous_piece_positions = {}
    game_moves = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = get_chessboard_corners(gray)

        if ret:
            warped = apply_perspective_transform(frame, corners)
            warped, squares = split_chessboard_into_squares(warped)
            if warped is None:
                continue

            current_piece_positions = detect_pieces(squares)
            movements = track_piece_movements(previous_piece_positions, current_piece_positions, game_moves)
            previous_piece_positions = current_piece_positions

            for (piece, old_alg, new_alg) in movements:
                print(f"{piece} moved from {old_alg} to {new_alg}")

            cv2.imshow("Warped Chessboard", warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_pgn(game_moves)

detect_chessboard_from_camera()