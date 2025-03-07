import cv2
import numpy as np

def get_chessboard_corners(gray, chessboard_size=(7, 7)):
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    return ret, corners

def apply_perspective_transform(frame, corners):
    corners = corners.reshape(-1, 2)
    src_pts = np.float32([corners[0], corners[6], corners[-1], corners[-7]])
    dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, M, (300, 300))

def split_chessboard_into_squares(warped):
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

def convert_position_to_algebraic(row, col):
    files = "abcdefgh"
    ranks = "87654321"
    return f"{files[col]}{ranks[row]}"