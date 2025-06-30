import cv2
import numpy as np
from sklearn.cluster import KMeans

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def pick_lines_evenly(lines, count=9):
    length = len(lines)
    if length <= count:
        return lines
    idxs = np.linspace(0, length-1, count)
    idxs = np.rint(idxs).astype(int)
    idxs = np.unique(idxs)
    idxs = [i for i in idxs if 0 <= i < length]
    return [lines[i] for i in idxs]

def filter_outliers_std(intersections, sigma=2.5):
    pts = np.array(intersections)
    if len(pts) < 2:
        return pts
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    lower = mean - sigma * std
    upper = mean + sigma * std
    mask = (pts[:, 0] >= lower[0]) & (pts[:, 0] <= upper[0]) & \
           (pts[:, 1] >= lower[1]) & (pts[:, 1] <= upper[1])
    return pts[mask]

def filter_outliers_percentile(intersections, low_pct=2, high_pct=98):
    pts = np.array(intersections)
    if len(pts) < 2:
        return pts
    xs = pts[:, 0]
    ys = pts[:, 1]
    x_min_clip = np.percentile(xs, low_pct)
    x_max_clip = np.percentile(xs, high_pct)
    y_min_clip = np.percentile(ys, low_pct)
    y_max_clip = np.percentile(ys, high_pct)
    mask = (xs >= x_min_clip) & (xs <= x_max_clip) & \
           (ys >= y_min_clip) & (ys <= y_max_clip)
    return pts[mask]

def detect_chessboard_corners(image):
    blockSize = 51
    C_value = 5
    hough_thresh = 100
    hough_minLen = 50
    hough_maxGap = 50
    min_line_length = 20
    max_line_length = 1000
    sigma_val = 2.5
    low_percentile = 2
    high_percentile = 98
    max_lines_per_dir = 20

    h_img, w_img = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize, C_value)
    invert = cv2.bitwise_not(thresh)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(invert, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.Canny(morph, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh,
                            minLineLength=hough_minLen, maxLineGap=hough_maxGap)
    if lines is None:
        print("[DEBUG] Brak linii z HoughLinesP.")
        return False, None
    lines = lines[:, 0, :]

    angles = []
    line_data = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        length = np.hypot(dx, dy)
        if min_line_length < length < max_line_length:
            angles.append([angle])
            line_data.append((x1, y1, x2, y2, angle, length))

    if len(line_data) < 2:
        print("[DEBUG] Za mało dobrych linii po filtrze długości.")
        return False, None

    # Grupowanie kątów
    kmeans = KMeans(n_clusters=2, n_init=10).fit(angles)
    group1 = [line_data[i] for i in range(len(line_data)) if kmeans.labels_[i] == 0]
    group2 = [line_data[i] for i in range(len(line_data)) if kmeans.labels_[i] == 1]

    def sort_and_pick(lines, key_fn):
        lines.sort(key=key_fn)
        return pick_lines_evenly(lines, max_lines_per_dir)

    def line_center_h(ln): return (ln[1] + ln[3]) / 2.0
    def line_center_v(ln): return (ln[0] + ln[2]) / 2.0

    group1 = sort_and_pick(group1, line_center_h)
    group2 = sort_and_pick(group2, line_center_v)

    def line_intersection(hln, vln):
        x1, y1, x2, y2, _, _ = hln
        x3, y3, x4, y4, _, _ = vln
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None
        Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (Px, Py)

    intersections = [line_intersection(h, v) for h in group1 for v in group2 if line_intersection(h, v)]
    if len(intersections) < 4:
        print("[DEBUG] Za mało punktów przecięcia.")
        return False, None

    pts_std = filter_outliers_std(intersections, sigma=sigma_val)
    if len(pts_std) < 4:
        print("[DEBUG] Za mało inliers po filter_outliers_std.")
        return False, None

    inliers = filter_outliers_percentile(pts_std, low_pct=low_percentile, high_pct=high_percentile)
    if len(inliers) < 4:
        print("[DEBUG] Za mało inliers po filter_outliers_percentile.")
        return False, None

    rect = cv2.minAreaRect(inliers.astype(np.float32))
    box = cv2.boxPoints(rect)
    corners_ordered = order_points(box)

    debug_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for ln in group1:
        cv2.line(debug_vis, (int(ln[0]), int(ln[1])), (int(ln[2]), int(ln[3])), (0, 255, 0), 2)
    for ln in group2:
        cv2.line(debug_vis, (int(ln[0]), int(ln[1])), (int(ln[2]), int(ln[3])), (0, 0, 255), 2)
    for pt in intersections:
        cv2.circle(debug_vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
    pts_int = corners_ordered.astype(int)
    cv2.polylines(debug_vis, [pts_int], True, (255, 255, 0), 3)
    cv2.imwrite("debug/final_lines.png", debug_vis)

    print("[DEBUG] bounding box:", corners_ordered)
    return True, corners_ordered
