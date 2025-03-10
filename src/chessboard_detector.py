import cv2
import numpy as np
import chess.pgn
import chess
from utils import apply_perspective_transform, split_chessboard_into_squares, convert_position_to_algebraic

def detect_chessboard(frame):
    """ Wykrywa szachownicę w obrazie, zwracając narożniki oraz przekształcony obraz """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcja narożników metodą Harris Corner Detection
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]  # Zaznacz narożniki na czerwono

    # Detekcja planszy metodą OpenCV
    chessboard_size = (7, 7)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        return True, corners
    return False, None

def detect_chessboard_from_camera():
    """ Główna pętla wykrywania szachownicy i śledzenia ruchów """
    cap = cv2.VideoCapture(0)
    previous_piece_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, corners = detect_chessboard(frame)
        if detected:
            warped = apply_perspective_transform(frame, corners)
            warped, squares = split_chessboard_into_squares(warped)

            cv2.imshow("Warped Chessboard", warped)

        cv2.imshow("Chessboard Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
