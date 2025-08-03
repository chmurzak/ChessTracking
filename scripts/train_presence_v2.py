import argparse, random, cv2, os, numpy as np, albumentations as A, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path

DATA_ROOT = Path("dataset/pieces")
MODEL_OUT = Path("models/presence_v2.h5")
IMG_SHAPE = (64, 64)             
EPOCHS    = 40
BATCH     = 64
SEED      = 42
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

AUG = A.Compose([
    A.RandomBrightnessContrast(0.35, 0.35, p=.9),
    A.HueSaturationValue(10, 15, 10, p=.7),
    A.RGBShift(10, 10, 10, p=.6),
    A.GaussNoise((5, 20), p=.3),
    A.ImageCompression(40,  p=.3),
])

def collect_paths():
    p, y = [], []
    for sq in ("white_squares", "black_squares"):
        for sub in (DATA_ROOT / sq).iterdir():
            if sub.is_dir():
                cls = 0 if sub.name == "empty" else 1
                files = list(sub.glob("*.png"))
                p.extend(files); y.extend([cls]*len(files))
    return np.array(p), np.array(y, "int8")

def load_imgs(paths, aug=None):
    arr = []
    for fp in paths:
        img = cv2.imread(str(fp))          
        if img is None: continue
        img = aug(image=img)["image"] if aug else img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab = cv2.resize(lab, IMG_SHAPE) / 255.
        arr.append(lab)
    return np.asarray(arr, "float32")

def oversample(X, y):
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    n = max(len(pos), len(neg))
    Xb = np.concatenate([X[np.random.choice(pos, n, True)],
                         X[np.random.choice(neg, n, True)]])
    yb = np.concatenate([np.ones(n,"float32"), np.zeros(n,"float32")])
    sh = np.random.permutation(len(Xb))
    return Xb[sh], yb[sh]

def baseline_cnn():
    inp = tf.keras.Input((*IMG_SHAPE,3))
    x = tf.keras.layers.Conv2D(32,3,activation='relu')(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64,3,activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128,3,activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inp,out)
    model.compile('adam','binary_crossentropy',metrics=['accuracy'])
    return model

if __name__ == "__main__":
    paths, labels = collect_paths()
    p_tr,p_val,y_tr,y_val = train_test_split(paths,labels,
                                             test_size=.20,
                                             stratify=labels,
                                             random_state=SEED)

    X_val = load_imgs(p_val)
    X_tr  = load_imgs(p_tr, AUG)
    X_tr, y_tr = oversample(X_tr, y_tr)

    ds_tr  = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr.reshape(-1,1)))
              .shuffle(4096).batch(BATCH).prefetch(2))
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val.reshape(-1,1)))\
              .batch(BATCH)

    model = baseline_cnn()
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=.5, patience=3, verbose=1),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1)
    ]

    model.fit(ds_tr,
              epochs=EPOCHS,
              validation_data=ds_val,
              callbacks=cbs,
              verbose=2)

    preds = (model.predict(X_val,256).squeeze() > 0.5).astype(int)
    print("CM [[TN,FP],[FN,TP]]\n", confusion_matrix(y_val, preds))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print("Zapisano", MODEL_OUT)
