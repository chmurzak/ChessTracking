import cv2
import numpy as np

def warp_perspective(image, corners, size=512):
    def order_corners(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    ordered = order_corners(corners)
    dst_pts = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    return warped

def split_board_into_squares(warped):
    squares = []
    h, w = warped.shape[:2]
    square_size = h // 8

    for row in range(8):
        for col in range(8):
            y1 = row * square_size
            y2 = (row + 1) * square_size
            x1 = col * square_size
            x2 = (col + 1) * square_size
            square = warped[y1:y2, x1:x2]
            squares.append(square)
    return squares
