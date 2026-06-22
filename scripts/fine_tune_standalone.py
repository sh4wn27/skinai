"""Standalone fine-tuning script — runs inside Modal container.

Called by fine_tune_modal.py via subprocess. No Modal dependencies.
Reads weights from /weights/, saves finetuned_*.h5 back to /weights/.
"""

import argparse
import gc
import io
import sys
import traceback

import numpy as np
import requests
import tf_keras as keras
from PIL import Image as PILImage

ISIC_SEARCH = "https://api.isic-archive.com/api/v2/images/search/"

CLASS_QUERIES = {
    "AK":   [("diagnosis_3", "Solar or actinic keratosis")],
    "BCC":  [("diagnosis_3", "Basal cell carcinoma")],
    "BKL":  [("diagnosis_2", "Benign epidermal proliferations")],
    "DF":   [("diagnosis_3", "Dermatofibroma")],
    "MEL":  [("diagnosis_2", "Malignant melanocytic proliferations (Melanoma)")],
    "NV":   [("diagnosis_3", "Nevus")],
    "SCC":  [("diagnosis_2", "Malignant epidermal proliferations")],
    "VASC": [
        ("diagnosis_3", "Hemangioma"),
        ("diagnosis_3", "Angiokeratoma"),
        ("diagnosis_3", "Pyogenic granuloma"),
    ],
}
CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "UNK", "VASC"]
CODE_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

_TRAIN_CLASS_COUNTS = {
    "AK": 867, "BCC": 3323, "BKL": 2624, "DF": 239,
    "MEL": 4522, "NV": 12875, "SCC": 628, "UNK": 0, "VASC": 253,
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fetch_samples(code, n, skip):
    samples = []
    cursor = None
    for field, value in CLASS_QUERIES[code]:
        remaining = n + skip
        while len(samples) < remaining:
            params = {"query": f'{field}:"{value}"', "limit": min(50, remaining)}
            if cursor:
                params["cursor"] = cursor
            r = requests.get(ISIC_SEARCH, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            samples.extend(data["results"])
            cursor = data.get("next")
            if cursor:
                cursor = cursor.split("cursor=")[-1].split("&")[0]
            if not cursor or not data["results"]:
                break
    return samples[skip: skip + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_file")
    ap.add_argument("--weights-dir", default="/weights")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--skip", type=int, default=55)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"Fine-tuning {args.model_file}", flush=True)
    print(f"per_class={args.per_class}  skip={args.skip}  epochs={args.epochs}", flush=True)
    print(f"{'='*60}\n", flush=True)

    model_path = f"{args.weights_dir}/{args.model_file}"
    model = keras.models.load_model(model_path, compile=False)
    _, H, W, _ = model.input_shape
    print(f"Loaded — input {H}×{W}", flush=True)

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-30:]:
        layer.trainable = True
    trainable = sum(p.numpy().size for p in model.trainable_weights)
    print(f"Trainable params: {trainable:,}", flush=True)

    # Download images
    X_uint8, y, class_counts = [], [], {}
    for code in [c for c in CLASSES if c != "UNK"]:
        print(f"Fetching {code}…", flush=True)
        samples = fetch_samples(code, args.per_class, args.skip)
        idx = CODE_TO_IDX[code]
        ok = 0
        for s in samples:
            try:
                raw = requests.get(s["files"]["full"]["url"], timeout=30).content
                img = PILImage.open(io.BytesIO(raw)).convert("RGB").resize((W, H))
                X_uint8.append(np.asarray(img, dtype=np.uint8))
                y.append(idx)
                ok += 1
            except Exception:
                continue
        class_counts[code] = ok
        print(f"  {code}: {ok}", flush=True)

    X = (np.array(X_uint8, dtype=np.float32) / 255.0 - MEAN) / STD
    del X_uint8; gc.collect()
    y = np.array(y, dtype=np.int32)
    print(f"\nTotal: {len(y)} images", flush=True)

    # LDAM margins
    C = 0.5 / max(1.0 / max(1, _TRAIN_CLASS_COUNTS[c]) ** 0.25
                  for c in CLASSES if c != "UNK")
    margins = np.array([
        C / max(1, _TRAIN_CLASS_COUNTS[c]) ** 0.25 if c != "UNK" else 0.0
        for c in CLASSES
    ], dtype=np.float32)
    print(f"Margins: { {c: f'{margins[i]:.3f}' for i, c in enumerate(CLASSES)} }", flush=True)

    def ldam_loss(y_true, y_pred):
        import tensorflow as tf
        y_true_int = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
        m = tf.gather(tf.constant(margins), y_true_int)
        oh = tf.one_hot(y_true_int, depth=len(CLASSES))
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred - oh * m[:, tf.newaxis])
        )

    freq = np.array([max(1, class_counts.get(c, 1)) for c in CLASSES], dtype=np.float32)
    cw_arr = np.minimum(1.0 / freq / (1.0 / freq).sum() * len(CLASSES), 10.0)
    class_weights = {i: float(cw_arr[i]) for i in range(len(CLASSES))}

    y_onehot = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1.0

    lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    drw = args.epochs // 2

    print(f"\nPhase 1: CE, {drw} epochs", flush=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y_onehot, batch_size=16, epochs=drw, class_weight=class_weights, shuffle=True, callbacks=[lr_cb], verbose=1)

    print(f"\nPhase 2: LDAM, {args.epochs - drw} epochs", flush=True)
    model.compile(optimizer=keras.optimizers.Adam(5e-5), loss=ldam_loss, metrics=["accuracy"])
    model.fit(X, y_onehot, batch_size=16, epochs=args.epochs - drw, class_weight=class_weights, shuffle=True, callbacks=[lr_cb], verbose=1)

    out = f"{args.weights_dir}/finetuned_{args.model_file}"
    model.save(out)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
