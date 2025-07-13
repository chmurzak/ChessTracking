import os, cv2, random, numpy as np, albumentations as A, tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2B0, preprocess_input)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path

DATA_ROOT = Path("dataset/pieces")
MODEL_OUT = Path("models/figure_color_v2.h5")
IMG_SHAPE = (96, 96)
EPOCHS    = 5                    
BATCH     = 32
SEED      = 42
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

AUG_TRAIN = A.Compose([
    A.RandomBrightnessContrast(0.40, 0.40, p=.9),
    A.HueSaturationValue(15, 25, 15, p=.7),
    A.RGBShift(10, 10, 10,        p=.6),
    A.ISONoise(color_shift=(0.01,0.05), p=.3),
    A.MotionBlur(3,               p=.2),
])

def iter_paths():
    """yield Path, label  (white = 1,  black = 0)"""
    for sq in ("white_squares", "black_squares"):
        for sub in (DATA_ROOT / sq).iterdir():
            if not sub.is_dir():
                continue
            cls = 1 if "white_" in sub.name else 0
            for fp in sub.glob("*.png"):
                yield fp, cls

def load_img(fp: Path, augment=None) -> np.ndarray:
    img = cv2.imread(str(fp))             
    if img is None:
        raise IOError(fp)
    if augment:
        img = augment(image=img)["image"]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    img = cv2.resize(img, IMG_SHAPE).astype("float32")
    img = preprocess_input(img)                   
    return img

def make_tf_dataset(file_paths, labels, training: bool):
    fp_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def _load(fp, lab):
        img = tf.numpy_function(
            lambda p: load_img(Path(p.decode()), AUG_TRAIN if training else None),
            [fp], tf.float32)
        img.set_shape((*IMG_SHAPE, 3))
        return img, tf.cast(lab, tf.float32)

    ds = fp_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(4096)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def build_model() -> tf.keras.Model:
    base = EfficientNetV2B0(include_top=False,
                            pooling="avg",
                            weights="imagenet",
                            input_shape=(*IMG_SHAPE, 3))
    for l in base.layers[:-30]:           
        l.trainable = False

    x   = tf.keras.layers.Dense(128, activation="relu")(base.output)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(base.input, out)
    model.compile(
        tf.keras.optimizers.Adam(1e-4),
        "binary_crossentropy",
        metrics=["accuracy"])
    return model

if __name__ == "__main__":
    paths, labels = zip(*list(iter_paths()))
    paths  = np.array([str(p) for p in paths]);  labels = np.array(labels, "int8")
    print(f"Found {len(paths)} tiles – white: {(labels==1).sum()}  black: {(labels==0).sum()}")

    p_tr, p_val, y_tr, y_val = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=SEED)

    idx_w, idx_b = np.where(y_tr==1)[0], np.where(y_tr==0)[0]
    n_max = max(len(idx_w), len(idx_b))
    over_idx = np.concatenate([
        np.random.choice(idx_w, n_max, replace=True),
        np.random.choice(idx_b, n_max, replace=True)])
    np.random.shuffle(over_idx)

    ds_train = make_tf_dataset(p_tr[over_idx], y_tr[over_idx], training=True)
    ds_val   = make_tf_dataset(p_val,          y_val,         training=False)

    model = build_model()

    cbs = []    

    model.fit(ds_train,
              epochs          = EPOCHS,
              validation_data = ds_val,
              callbacks       = cbs)

    probs = model.predict(ds_val, verbose=0).squeeze()
    preds = (probs > 0.5).astype(int)
    print("Confusion Matrix [[TN,FP],[FN,TP]]\n",
          confusion_matrix(y_val, preds))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print("✓ zapisano", MODEL_OUT)
