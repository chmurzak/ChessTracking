from chessboard_detector import detect_chessboard_from_camera
from classification import detect_pieces_yolo
from tracking import ChessPieceTracker
import cv2

if __name__ == "__main__":
    print("Starting ChessVision...")

    tracker = ChessPieceTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, detected_pieces = detect_pieces_yolo(frame)
        tracked_pieces = tracker.update(frame, detected_pieces)

        for piece_id, bbox in tracked_pieces:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {piece_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Chess Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
