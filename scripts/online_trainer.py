import os, cv2, random, threading, time, json
from collections import deque
from queue import Queue
from pathlib import Path

import torch, timm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# ---- konfiguracja ----
CLASS_NAMES  = ["queen", "rook", "knight", "bishop"]
LIVE_DIR     = Path("live_pool")
POOL_MAX     = 120        # per klasa
BATCH_UPDATE = 256        # ile nowych próbek wyzwala update
TRAIN_STEPS  = 4          # kroków opt w każdej aktualizacji
LR           = 1e-5
IMG_SIZE     = 64
REPLAY       = 200        # ile starych przykładów dorzucamy
MODEL_DIR    = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PTH    = MODEL_DIR / "current.pth"
MODEL_TS     = MODEL_DIR / "current.ts"

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def _init_dirs():
    LIVE_DIR.mkdir(exist_ok=True)
    for cls in CLASS_NAMES:
        (LIVE_DIR / cls).mkdir(exist_ok=True)

_init_dirs()

class Net(nn.Module):
    def __init__(self, n_cls: int = 4):
        super().__init__()
        self.back = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        for p in self.back.parameters():
            p.requires_grad_(False)               
        self.head = nn.Sequential(
            nn.Linear(self.back.num_features, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        return self.head(self.back(x))

model = Net(len(CLASS_NAMES))
if MODEL_PTH.exists():
    model.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
model.eval()

_head_params = [p for p in model.head.parameters() if p.requires_grad]
opt  = optim.AdamW(_head_params, lr=LR)
crit = nn.CrossEntropyLoss()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

lock        = threading.Lock()            
train_queue = Queue()                      
_new_count  = 0                            

_pool = {cls: deque(maxlen=POOL_MAX) for cls in CLASS_NAMES}


def save_tile(tile_bgr, cls_name: str, force: bool = False):

    if cls_name not in CLASS_NAMES:
        return

    global _new_count
    ts = time.time_ns()
    fname = f"{ts}.png"
    out_path = LIVE_DIR / cls_name / fname
    cv2.imwrite(str(out_path), tile_bgr)
    _pool[cls_name].append(out_path)

    _new_count += 1
    if _new_count >= BATCH_UPDATE or force:
        try:
            train_queue.put_nowait(1)
            _new_count = 0
        except:
            pass  # kolejka pełna – update już zaplanowany


def _collect_replay(base_dir: Path):
    paths = []
    for cls in CLASS_NAMES:
        for sq in ("black_squares", "white_squares"):
            for col in ("black", "white"):
                d = base_dir / sq / f"{col}_{cls}"
                if d.is_dir():
                    paths += list((d).glob("*.png"))
    random.shuffle(paths)
    return paths[:REPLAY]

_base_replay_paths = _collect_replay(Path("dataset/pieces"))
_base_label_map = {p: CLASS_NAMES.index(p.parent.name.split("_", 1)[-1])
                   for p in _base_replay_paths}

class _DS(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if img is None:
            raise RuntimeError(f"Corrupt image: {p}")
        tensor = _transform(img)
        parent = p.parent.name       
        cls_name = parent.split("_", 1)[-1]
        label = CLASS_NAMES.index(cls_name)
        return tensor, label

def _trainer_loop():
    while True:
        train_queue.get()        
        live_paths = []
        for cls in CLASS_NAMES:
            live_paths += list(_pool[cls])
        paths = live_paths + _base_replay_paths
        ds = _DS(paths)
        ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        model.train(); step = 0
        for x, y in ld:
            opt.zero_grad(); out = model(x); loss = crit(out, y)
            loss.backward(); opt.step()
            step += 1
            if step >= TRAIN_STEPS:
                break
        model.eval()
        with lock:
            torch.save(model.state_dict(), MODEL_PTH)
            torch.jit.trace(model, dummy).save(MODEL_TS)
        print(f"[online_trainer] head updated (loss={loss.item():.4f})")

threading.Thread(target=_trainer_loop, daemon=True).start()