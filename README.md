# ChessTracking

A system for recording and broadcasting chess games based purely on image analysis from a camera.  
The goal of this project is to provide a **fast and low-cost backend** for generating games in **PGN format**, without the need for expensive electronic boards.

---

## How it works

1. **Video source** – ideally from a smartphone using the **IP Webcam** app (Android).  
   After launching the app, connect to the local stream at `http://<IP:PORT>/video`.

2. **Camera setup** – place the camera steadily above the chessboard.

3. **Calibration** – press the `K` key to detect the corners of the board, set the orientation, and save the 8×8 grid.

4. **Move registration** – after each move, press the `D` key.  
   The system checks **only the fields that changed**, which makes it **very fast and lightweight on CPU**.

5. **Final output** – the game is recorded in PGN format in: **debug/game.pgn**

---

## Repository structure

- `src/` – main source code (Python + OpenCV + lightweight ML models).
- `models/` – contains three pre-trained models for each classification/detection task.
- `runs/` – logs of model training and evaluation ([examples here](https://github.com/chmurzak/ChessTracking/tree/main/runs)).
- `start_position.json` – description of the initial chess position.
- `requirements.txt` – Python dependencies.
- `.gitignore`, `README.md` – auxiliary files.

---

## Modes

- **K (Calibration)**  
Detects board corners, applies perspective transform, and splits the board into 64 tiles.  
At this stage, the system establishes orientation and saves the base state.

- **D (Detection)**  
Tracks changes after each move using lightweight classifiers:  
- presence (empty / occupied),  
- piece color (white / black).  
- promotion model (used only when a pawn reaches the last rank, to recognize which piece it was promoted to: queen, rook, bishop, or knight). 

This enables **near real-time move registration**.

---

## Notes

- The project currently works only as a **backend** – no graphical frontend yet.  
- Optimized for speed – runs smoothly on a standard laptop CPU.  
- Key logs and outputs are stored in the `debug/` directory.

---

## Author

This repository was developed as part of a Master’s thesis at AGH University of Science and Technology, focused on advanced image processing methods applied to chess game recording and analysis.
