import re, os, datetime as dt
import chess, chess.pgn

MOVES_TXT = "debug/moves.txt"
OUT_PGN   = "debug/game.pgn"


def _parse_moves(moves_path=MOVES_TXT):
    if not os.path.exists(moves_path):
        return []

    moves = []
    with open(moves_path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r":\s*([a-h][1-8]-?[a-h][1-8])", line)
            if not m:
                continue
            uci = m.group(1).replace("-", "")
            if not moves or moves[-1] != uci:
                moves.append(uci)
    return moves


def _build_game(moves_uci):
    board = chess.Board()
    game  = chess.pgn.Game()
    game.headers.update({
        "Event": "ChessVision Live",
        "Site" : "Local",
        "Date" : dt.date.today().strftime("%Y.%m.%d"),
        "White": "White",
        "Black": "Black",
    })
    node = game

    for uci in moves_uci:
        try:
            move = board.push_uci(uci)   
        except ValueError:
            break
        node = node.add_variation(move)

    return game


def write_pgn(moves_uci, out_path=OUT_PGN):
    game = _build_game(moves_uci)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")


def update_pgn():
    moves = _parse_moves()
    if moves:                 
        write_pgn(moves)
