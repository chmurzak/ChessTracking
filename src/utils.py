from sklearn.cluster import DBSCAN
import numpy as np

def segment_chessboard_dbscan(corners):
    """ Używa DBSCAN do grupowania narożników w 64 pola szachownicy """
    corners = corners.reshape(-1, 2)
    clustering = DBSCAN(eps=15, min_samples=2).fit(corners)
    labels = clustering.labels_

    unique_labels = set(labels)
    squares = []
    for label in unique_labels:
        if label == -1:  # Ignorujemy outliery
            continue
        points = corners[labels == label]
        x, y = np.mean(points, axis=0)
        squares.append((int(x), int(y)))

    return squares
