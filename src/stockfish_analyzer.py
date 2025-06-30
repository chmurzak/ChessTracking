from stockfish import Stockfish

stockfish = Stockfish("path/to/stockfish.exe") 

def analyze_position(fen):
    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()
    return best_move
