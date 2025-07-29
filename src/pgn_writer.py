import os, re, datetime as dt
import chess, chess.pgn

MOVES_TXT = "debug/moves.txt"
OUT_PGN   = "debug/game.pgn"

CASTLE_MAP = {
    "e1g1": "O-O",  "e1c1": "O-O-O",
    "e8g8": "O-O",  "e8c8": "O-O-O",
}

def uci_to_san(uci: str) -> str:
    return CASTLE_MAP[uci]

def _parse_moves(moves_path: str = MOVES_TXT) -> list[str]:
    if not os.path.exists(moves_path):
        return []

    moves: list[str] = []
    with open(moves_path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r":\s*([a-h][1-8]-?[a-h][1-8][qQrRnNbB]?)", line)
            if not m:
                continue
            uci = m.group(1).replace("-", "")

            if moves and moves[-1][:4] == uci[:4]:
                if len(uci) > len(moves[-1]):
                    moves[-1] = uci
            else:
                moves.append(uci)
    return moves

def _build_game(moves_uci: list[str]) -> chess.pgn.Game:
    board = chess.Board()
    game  = chess.pgn.Game()
    game.headers.update({
        "Event": "ChessVision Live",
        "Site" : "Local",
        "Date" : dt.date.today().strftime("%Y.%m.%d"),
        "White": "White",
        "Black": "Black",
        "Result": "*",
    })

    node = game
    for uci in moves_uci:
        try:
            move_obj = chess.Move.from_uci(uci)
            san = uci_to_san(uci) if uci in CASTLE_MAP else board.san(move_obj)
            board.push(move_obj)
        except ValueError:
            break

        node = node.add_variation(move_obj, comment=san)

    return game

def write_pgn(moves_uci: list[str], out_path: str = OUT_PGN) -> None:
    game = _build_game(moves_uci)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")

def update_pgn() -> None:
    moves = _parse_moves()
    if moves:
        write_pgn(moves)
