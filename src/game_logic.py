import chess.pgn
import chess

def save_pgn(game_moves, filename="game.pgn"):
    """ Tworzy pełny zapis PGN gry szachowej """
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessVision AI Game"
    game.headers["Site"] = "Local"
    game.headers["Date"] = "2024.03.07"
    game.headers["Round"] = "1"
    game.headers["White"] = "Player1"
    game.headers["Black"] = "Player2"

    node = game
    for move in game_moves:
        chess_move = chess.Move.from_uci(move)
        node = node.add_variation(chess_move)

    with open(filename, "w") as pgn_file:
        pgn_file.write(str(game))
