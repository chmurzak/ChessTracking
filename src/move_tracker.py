def _vec(sq: str):
    """Zwraca (file, rank) :  a-h → 0-7,  rank → int."""
    return ord(sq[0]) - 97, int(sq[1])


def detect_move_full(prev_simple: dict,
                     curr_simple: dict,
                     prev_full  : dict):

    disappeared, appeared = [], []

    # 1. Różnice między klatkami
    for sq in prev_simple:
        if prev_simple[sq] == curr_simple[sq]:
            continue
        if prev_simple[sq] != "empty":
            disappeared.append(sq)
        if curr_simple[sq] != "empty":
            appeared.append(sq)

    # 2a. Roszady  (2 znikają, 2 pojawiają się)
    if len(disappeared) == 2 and len(appeared) == 2:
        dset, aset = set(disappeared), set(appeared)
        fp = {sq: prev_full.get(sq, "") for sq in disappeared}
        isK = lambda c: c and c.lower().endswith("k")
        isR = lambda c: c and c.lower().endswith("r")

        if isK(fp.get("e1")) and isR(fp.get("h1")) and dset == {"e1", "h1"} and aset == {"f1", "g1"}:
            return "e1g1", "K"
        if isK(fp.get("e1")) and isR(fp.get("a1")) and dset == {"e1", "a1"} and aset == {"c1", "d1"}:
            return "e1c1", "K"
        if isK(fp.get("e8")) and isR(fp.get("h8")) and dset == {"e8", "h8"} and aset == {"f8", "g8"}:
            return "e8g8", "K"
        if isK(fp.get("e8")) and isR(fp.get("a8")) and dset == {"e8", "a8"} and aset == {"c8", "d8"}:
            return "e8c8", "K"

    # 2b / 2c.  (2 znikają → 1 się pojawia)
    if len(disappeared) == 2 and len(appeared) == 1:
        to_sq           = appeared[0]
        tcol, trow      = _vec(to_sq)

        pawn_from = next(
            (sq for sq in disappeared
             if abs(_vec(sq)[0] - tcol) == 1
             and abs(_vec(sq)[1] - trow) == 1
             and prev_full[sq].lower().endswith("p")),  
            None
        )

        if pawn_from:
            captured = next(sq for sq in disappeared if sq != pawn_from)

            # ── promocja z biciem
            if to_sq[1] in ("1", "8"):
                promo = "q" if prev_full[pawn_from].islower() else "Q"
                return pawn_from + to_sq + promo.lower(), promo

            # ── en-passant  (docelowo zawsze 3 lub 6)
            if to_sq[1] in ("3", "6") \
               and prev_full[captured].lower().endswith("p"):  
                return pawn_from + to_sq, "ep"

    # 3. 1 znikło → 1 się pojawiło
    if len(disappeared) == 1 and len(appeared) == 1:
        frm, to = disappeared[0], appeared[0]
        piece   = prev_full.get(frm, "")

        # promocja bez bicia
        if piece.lower().endswith("p") and to[1] in ("1", "8"):
            promo = "q" if piece.islower() else "Q"
            return frm + to + promo.lower(), promo

        # zwykły ruch / bicie
        return frm + to, piece

    return None, None


def apply_move_to_full(prev_full: dict,
                       move_uci : str,
                       piece_code: str):

    if not move_uci or len(move_uci) < 4:
        return

    frm, to = move_uci[:2], move_uci[2:4]

    # 1. Roszady
    if piece_code == "K" and move_uci in ("e1g1", "e1c1", "e8g8", "e8c8"):
        if move_uci == "e1g1":
            prev_full.update({"e1": "empty", "g1": "K", "h1": "empty", "f1": "R"})
        elif move_uci == "e1c1":
            prev_full.update({"e1": "empty", "c1": "K", "a1": "empty", "d1": "R"})
        elif move_uci == "e8g8":
            prev_full.update({"e8": "empty", "g8": "K", "h8": "empty", "f8": "R"})
        else:
            prev_full.update({"e8": "empty", "c8": "K", "a8": "empty", "d8": "R"})
        return

    # 2. Promocja
    if len(move_uci) == 5 and move_uci[-1].lower() == "q":
        prev_full[frm] = "empty"
        prev_full[to]  = piece_code     
        return

    # 3. En-passant
    if piece_code == "ep":
        pawn_code     = prev_full[frm]
        captured_sq   = to[0] + frm[1]
        prev_full[frm]        = "empty"
        prev_full[captured_sq] = "empty"
        prev_full[to]         = pawn_code
        return

    # 4. Zwykły ruch / bicie
    prev_full[frm] = "empty"
    prev_full[to]  = piece_code
