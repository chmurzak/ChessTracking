from stockfish import Stockfish

stockfish = Stockfish("path/to/stockfish.exe") # dodać ścieżkę

def analyze_position(fen):
    """ Analizuje pozycję i podaje najlepszy ruch """
    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()
    return best_move
