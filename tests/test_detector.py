import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src_old_canny_houghlines.chessboard_detector import detect_chessboard_corners
from src.cv_utils import warp_perspective, split_board_into_squares

image_path = "dataset/training_set/1B2b3-Kp6-8-8-2k5-8-8-8.JPG"
image = cv2.imread(image_path)

if image is None:
    print(f"Nie znaleziono obrazu: {image_path}")
    exit()

found, corners = detect_chessboard_corners(image)

if found:
    print("Szachownica znaleziona. Narożniki:", corners)
    warped = warp_perspective(image, corners)
    cv2.imwrite("debug/warped_output.png", warped)
    print("Rozmiar po przekształceniu:", warped.shape)

    squares = split_board_into_squares(warped)
    for i, sq in enumerate(squares):
        cv2.imwrite(f"debug/square_{i}.png", sq)

    print("Szachownica podzielona na 64 pola.")
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nie wykryto szachownicy.")
