import chess

def rc_to_algebraic(row, col):

    file = chr(col + 97)       
    rank = str(8 - row)        
    return file + rank

def board_dict_to_8x8(board_dict):

    files = "abcdefgh"
    ranks = "87654321"
    board_8x8 = []
    for r in ranks:
        row = []
        for f in files:
            row.append(board_dict.get(f + r, "empty"))
        board_8x8.append(row)
    return board_8x8

class BoardState:

    def __init__(self):
        self.current_state = None  
        self.prev_state = None
        self.chess_board = chess.Board()

    def update_state(self, new_8x8):

        if self.current_state is None:
            self.current_state = new_8x8
            return None

        self.prev_state = self.current_state
        self.current_state = new_8x8

        changes = []
        for r in range(8):
            for c in range(8):
                old_val = self.prev_state[r][c]
                new_val = self.current_state[r][c]
                if old_val != new_val:
                    changes.append((r, c, old_val, new_val))

        if len(changes) == 2:
            (r1, c1, o1, n1) = changes[0]
            (r2, c2, o2, n2) = changes[1]

            start_row, start_col = None, None
            end_row, end_col = None, None

            if o1 != "empty" and n1 == "empty":
                start_row, start_col = r1, c1
                end_row, end_col = r2, c2
            else:
                start_row, start_col = r2, c2
                end_row, end_col = r1, c1

            move_uci = rc_to_algebraic(start_row, start_col) + rc_to_algebraic(end_row, end_col)

            move_obj = chess.Move.from_uci(move_uci)
            if move_obj in self.chess_board.legal_moves:
                self.chess_board.push(move_obj)
                return move_uci
            else:
                return move_uci  

        return None

def detect_simple_move(prev_dict, curr_dict):
    disappeared = []
    appeared = []
    for sq in prev_dict:
        if prev_dict[sq] != curr_dict[sq]:
            if prev_dict[sq] != "empty" and curr_dict[sq] == "empty":
                disappeared.append(sq)
            elif curr_dict[sq] != "empty" and prev_dict[sq] == "empty":
                appeared.append(sq)
    if len(disappeared) == 1 and len(appeared) == 1:
        return f"{disappeared[0]}-{appeared[0]}", prev_dict[disappeared[0]]
    return None, None
