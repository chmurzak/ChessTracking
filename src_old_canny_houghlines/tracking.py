import sys
import torch
import cv2
import numpy as np

sys.path.append("FairMOT/src/lib")

from FairMOT.src.lib.tracker.multitracker import JDETracker
from FairMOT.src.lib.tracking_utils.timer import Timer
from FairMOT.src.lib.tracking_utils.log import logger
from FairMOT.src.lib.opts import opts

def convert_detections_for_tracker(detected_pieces, conf_thres=0.4):
    detections_for_tracker = []
    for det in detected_pieces:
        x1, y1, x2, y2 = det["bbox"]
        w = x2 - x1
        h = y2 - y1
        detections_for_tracker.append([x1, y1, w, h, 1.0])

    if len(detections_for_tracker) == 0:
        return np.zeros((0,5), dtype=np.float32)
    return np.array(detections_for_tracker, dtype=np.float32)

class ChessPieceTracker:

    def __init__(self, model_path="models/fairmot_model.pth", conf_thres=0.4, track_buffer=30):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.track_buffer = track_buffer

        self.opt = opts().init()
        self.opt.load_model = self.model_path
        self.opt.conf_thres = self.conf_thres
        self.opt.track_buffer = self.track_buffer

        self.tracker = JDETracker(self.opt, frame_rate=30)

    def update(self, frame, detected_pieces):
        timer = Timer()
        timer.tic()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_frame).float().permute(2,0,1).unsqueeze(0) / 255.0

        xywhs = convert_detections_for_tracker(detected_pieces, self.conf_thres)

        online_targets = self.tracker.update(img_tensor, xywhs)

        tracked_pieces = []
        for t in online_targets:
            piece_id = t.track_id
            x1, y1, x2, y2 = t.tlbr
            tracked_pieces.append((piece_id, (int(x1), int(y1), int(x2), int(y2))))

        timer.toc()
        fps = 1./max(1e-5, timer.average_time)
        logger.info(f"FairMOT Tracking FPS: {fps:.2f}")

        return tracked_pieces
