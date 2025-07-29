def _vec(square: str) -> tuple[int, int]:
    return ord(square[0]) - 97, int(square[1])


def detect_move_full(prev_simple: dict,
                     curr_simple: dict,
                     prev_full  : dict):

    disappeared, appeared = [], []

    for sq in prev_simple:
        if prev_simple[sq] == curr_simple[sq]:
            continue
        if prev_simple[sq] != "empty":
            disappeared.append(sq)
        if curr_simple[sq] != "empty":
            appeared.append(sq)

    # ROSZADY (2 → 2)
    if len(disappeared) == 2 and len(appeared) == 2:
        dset, aset = set(disappeared), set(appeared)
        fp = {sq: prev_full.get(sq, "") for sq in disappeared}
        isK = lambda c: c.lower().endswith("k") if c else False
        isR = lambda c: c.lower().endswith("r") if c else False

        if isK(fp.get("e1")) and isR(fp.get("h1")) and dset == {"e1", "h1"} and aset == {"f1", "g1"}:
            return "e1g1", "K"
        if isK(fp.get("e1")) and isR(fp.get("a1")) and dset == {"e1", "a1"} and aset == {"c1", "d1"}:
            return "e1c1", "K"
        if isK(fp.get("e8")) and isR(fp.get("h8")) and dset == {"e8", "h8"} and aset == {"f8", "g8"}:
            return "e8g8", "K"
        if isK(fp.get("e8")) and isR(fp.get("a8")) and dset == {"e8", "a8"} and aset == {"c8", "d8"}:
            return "e8c8", "K"

    # BICIE SKOŚNE, EN-PASSANT, PROMOCJA Z BICIEM
    if len(disappeared) == 2 and len(appeared) == 1:
        to_sq         = appeared[0]
        tcol, trow    = _vec(to_sq)

        pawn_from = next(
            (sq for sq in disappeared
             if abs(_vec(sq)[0] - tcol) == 1         
             and abs(_vec(sq)[1] - trow) == 1
             and prev_full[sq].lower().endswith("p")),  
            None
        )

        if pawn_from:
            captured = next(sq for sq in disappeared if sq != pawn_from)

            if to_sq[1] in ("1", "8"):
                promo_map = {"queen": "q", "rook": "r", "knight": "n", "bishop": "b"}
                label     = curr_simple[to_sq]              
                p         = promo_map.get(label, "q")       # domyślnie hetman
                if prev_full[pawn_from].isupper():          
                    p = p.upper()
                return pawn_from + to_sq + p.lower(), p

            # — en-passant 
            if to_sq[1] in ("3", "6") and prev_full[captured].lower().endswith("p"):
                return pawn_from + to_sq, prev_full[pawn_from]

            # — zwykłe bicie pionem (bez promocji)
            return pawn_from + to_sq, prev_full[pawn_from]

    # 3 RUCH (1 → 1)  -- razem z promocją bez bicia
    if len(disappeared) == 1 and len(appeared) == 1:
        frm, to = disappeared[0], appeared[0]
        piece   = prev_full.get(frm, "")

        # promocja bez bicia
        if piece.lower().endswith("p") and to[1] in ("1", "8"):
            promo_map = {"queen": "q", "rook": "r", "knight": "n", "bishop": "b"}
            label     = curr_simple[to]                     
            p         = promo_map.get(label, "q")
            if piece.isupper():                             
                p = p.upper()
            return frm + to + p.lower(), p

        # zwykły ruch / bicie
        return frm + to, piece

    return None, None


def apply_move_to_full(board_full: dict,
                       move_uci   : str,
                       piece_code : str):

    if not move_uci or len(move_uci) < 4:
        return

    frm, to = move_uci[:2], move_uci[2:4]

    # 1 ─ Roszady
    if piece_code == "K" and move_uci in ("e1g1", "e1c1", "e8g8", "e8c8"):
        if   move_uci == "e1g1": board_full.update({"e1":"empty","g1":"K","h1":"empty","f1":"R"})
        elif move_uci == "e1c1": board_full.update({"e1":"empty","c1":"K","a1":"empty","d1":"R"})
        elif move_uci == "e8g8": board_full.update({"e8":"empty","g8":"K","h8":"empty","f8":"R"})
        else:                   board_full.update({"e8":"empty","c8":"K","a8":"empty","d8":"R"})
        return

    # 2 ─ Promocja
    if len(move_uci) == 5 and move_uci[-1].lower() in ("q", "r", "n", "b"):
        board_full[frm] = "empty"
        board_full[to]  = piece_code     
        return

    # 3 ─ Automatyczne bicie w przelocie 
    if piece_code.lower().endswith("p"):
        col_diff = abs(ord(frm[0]) - ord(to[0]))
        row_diff = abs(int(frm[1]) - int(to[1]))
        captured_sq = to[0] + frm[1]              
        if col_diff == 1 and row_diff == 1 and board_full.get(to) == "empty":
            if board_full.get(captured_sq, "").lower().endswith("p"):
                board_full[frm]         = "empty"
                board_full[captured_sq] = "empty"
                board_full[to]          = piece_code
                return

    # 4 ─ Zwykły ruch lub bicie
    board_full[frm] = "empty"
    board_full[to]  = piece_code
