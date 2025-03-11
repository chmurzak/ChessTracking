import sys
import torch
import cv2
import numpy as np
from FairMOT.src.lib.tracker.multitracker import JDETracker
from FairMOT.src.lib.tracking_utils.timer import Timer
from FairMOT.src.lib.tracking_utils.log import logger
from FairMOT.src.lib.opts import opts  # Konfiguracja modelu

sys.path.append("FairMOT/src/lib")

class ChessPieceTracker:
    """Klasa do śledzenia ruchów figur szachowych przy użyciu FairMOT."""

    def __init__(self, model_path="models/fairmot_model.pth", conf_thres=0.4, track_buffer=30):
        """Inicjalizuje tracker na podstawie modelu FairMOT."""
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.track_buffer = track_buffer

        # Inicjalizacja konfiguracji i modelu FairMOT
        self.opt = opts().init()
        self.tracker = JDETracker(self.opt, frame_rate=30)

    def update(self, frame, detected_pieces):
        """Aktualizuje śledzenie figur na podstawie aktualnej klatki i wykrytych obiektów."""
        timer = Timer()
        timer.tic()

        # Przetwarzanie klatki do formatu modelu FairMOT
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Uruchomienie trackera
        online_targets = self.tracker.update(img_tensor, detected_pieces)

        # Konwersja wyników
        tracked_pieces = []
        for target in online_targets:
            piece_id = target.track_id
            bbox = target.tlbr
            tracked_pieces.append((piece_id, bbox))

        timer.toc()
        logger.info(f"Tracking FPS: {1. / max(1e-5, timer.average_time):.2f}")

        return tracked_pieces
