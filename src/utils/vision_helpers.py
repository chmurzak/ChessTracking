import cv2, torch
from torchvision import transforms
from functools import lru_cache

CLASS_NAMES = ["queen", "rook", "knight", "bishop"]
MODEL_PATH  = "models/chess_piece_classifier.ts"
QUEEN_BIAS  = 1.05         

@lru_cache()                
def _load_model():
    model = torch.jit.load(MODEL_PATH).eval()
    tfm   = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    return model, tfm

def square_color(coord: str) -> str:
    file, rank = ord(coord[0]) - 97, int(coord[1]) - 1
    return "white_squares" if (file + rank) % 2 else "black_squares"

@torch.inference_mode()
def classify_promotion(tile_bgr):
    model, tfm = _load_model()
    tens   = tfm(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    logits = model(tens)
    logits[:, 0] *= QUEEN_BIAS         
    probs  = torch.softmax(logits, 1)
    idx    = int(torch.argmax(probs, 1))
    conf   = float(probs[0, idx])       
    return CLASS_NAMES[idx], conf
