def detect_move_full(prev_simple: dict,
                     curr_simple: dict,
                     prev_full : dict):

    disappeared, appeared = [], []

    for sq in prev_simple:
        if prev_simple[sq] == curr_simple[sq]:
            continue
        if prev_simple[sq] != "empty" and curr_simple[sq] == "empty":
            disappeared.append(sq)
        elif prev_simple[sq] == "empty" and curr_simple[sq] != "empty":
            appeared.append(sq)

    if len(disappeared) == 1 and len(appeared) == 1:
        frm, to = disappeared[0], appeared[0]
        piece   = prev_full.get(frm, "unk")     
        return frm + to, piece

    return None, None


def apply_move_to_full(prev_full: dict,
                       move_uci : str,
                       piece_code: str):

    if not move_uci or len(move_uci) != 4:
        return

    frm, to = move_uci[:2], move_uci[2:]
    prev_full[frm] = "empty"
    prev_full[to]  = piece_code