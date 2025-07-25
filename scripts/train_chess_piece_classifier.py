import os, cv2, torch, timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_DIR    = "dataset/pieces"
BATCH_SIZE  = 32
NUM_EPOCHS  = 8
LR          = 3e-4
CLASS_NAMES = ["queen", "rook", "knight", "bishop"]
ALLOWED_EXT = (".png", ".jpg", ".jpeg")
QUEEN_BIAS  = 1.5             

def load_data(root):
    paths, labels = [], []
    for lbl, cls in enumerate(CLASS_NAMES):
        for square in ("black_squares", "white_squares"):
            for colour in ("black", "white"):
                d = os.path.join(root, square, f"{colour}_{cls}")
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    p = os.path.join(d, f)
                    if os.path.isfile(p) and p.lower().endswith(ALLOWED_EXT):
                        if cv2.imread(p) is not None:   
                            paths.append(p); labels.append(lbl)
    print(f"{len(paths)} poprawnych obrazów")
    return paths, labels

class ChessPieceDataset(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths, self.labels, self.tfm = paths, labels, tfm
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
        return self.tfm(img), self.labels[idx]

def safe_collate(batch):
    imgs, lbls = zip(*batch)
    return torch.stack(imgs), torch.tensor(lbls)

train_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),           
    transforms.GaussianBlur(3, sigma=(0.1,1.0)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
val_tfm = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

paths, labels = load_data(DATA_DIR)
tr_p, val_p, tr_l, val_l = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)
train_loader = DataLoader(
    ChessPieceDataset(tr_p, tr_l, train_tfm),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=safe_collate, num_workers=0
)
val_loader = DataLoader(
    ChessPieceDataset(val_p, val_l, val_tfm),
    batch_size=BATCH_SIZE, shuffle=False, collate_fn=safe_collate, num_workers=0
)

class ChessPieceClassifier(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0",
                                          pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, n_cls)
        )
    def forward(self, x):  return self.head(self.backbone(x))

model = ChessPieceClassifier(len(CLASS_NAMES))
opt   = optim.AdamW(model.parameters(), lr=LR)
crit  = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_ok = tot = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            if train: opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            if train: loss.backward(); opt.step()
            tot_loss += loss.item()*x.size(0)
            tot_ok   += (out.argmax(1)==y).sum().item()
            tot      += y.size(0)
    return tot_loss/tot, 100*tot_ok/tot

for ep in range(1, NUM_EPOCHS+1):
    tl, ta = run_epoch(train_loader, True)
    vl, va = run_epoch(val_loader, False)
    print(f"[{ep}/{NUM_EPOCHS}] "
          f"train {ta:5.1f}% / val {va:5.1f}% (loss {tl:.4f}/{vl:.4f})")

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in val_loader:
        logits = model(x)
        logits[:,0] *= QUEEN_BIAS    
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(1).tolist())

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion matrix validation set")
plt.tight_layout(); plt.savefig("confusion_matrix.png")
print("Zapisano confusion_matrix.png")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/chess_piece_classifier.pth")
print("Model .pth zapisany")

example = torch.randn(1, 3, 64, 64)
torch.jit.trace(model, example).save("models/chess_piece_classifier.ts")
print("TorchScript zapisany")

