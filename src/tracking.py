from fairmot_tracker import Tracker

tracker = Tracker(model_path="models/fairmot_model.pth")

def track_pieces(frame, detected_pieces):
    """ Śledzi ruchy figur na podstawie FairMOT """
    track_results = tracker.update(detected_pieces)
    return track_results
