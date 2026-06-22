"""Fine-tune the EfficientNet ensemble on class-balanced ISIC data using LDAM loss.

Runs as a Modal job on T4 GPU. Reads existing .h5 weights from the skinai-weights
volume, downloads balanced ISIC training data, fine-tunes the top layers of each
model, and writes updated weights back to the volume.

Usage:
    modal run scripts/fine_tune_modal.py          # fine-tune all 3 models
    modal run scripts/fine_tune_modal.py --model efficientnet_b4.h5  # one model

What this fixes:
    - DF/AK/SCC had 0% recall because the backbone never learned them (NV dominated
      training data at ~65%). LDAM loss penalises misclassifications of rare classes
      more heavily, and balanced sampling ensures equal class exposure during fine-tuning.

Strategy:
    - Freeze all base EfficientNet layers; unfreeze the top 30 layers + classifier head.
    - Download 120 images/class from ISIC (skipping first 55 to avoid eval/calibration overlap).
    - LDAM loss: modified cross-entropy where logit for true class y is reduced by
      margin Δ_y = C / n_y^{1/4}. This pushes the decision boundary away from rare classes.
    - DRW (Deferred Re-Weighting): first half of training uses standard CE + balanced sampling;
      second half switches to LDAM + class weights. This prevents early training collapse.
    - Learning rate 1e-4 with cosine decay; batch size 16; 20 epochs per model.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import modal

app = modal.App("skinai-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow==2.20.0",
        "tf-keras==2.20.0",
        "pillow==11.2.1",
        "numpy==2.1.3",
        "requests==2.32.3",
        "scikit-learn==1.5.2",
    )
)

weights_volume = modal.Volume.from_name("skinai-weights", create_if_missing=True)

ISIC_SEARCH = "https://api.isic-archive.com/api/v2/images/search/"

CLASS_QUERIES: dict[str, list[tuple[str, str]]] = {
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

# Alphabetical — must match the order the original model was trained with
CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "UNK", "VASC"]
CODE_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Approximate class frequencies in the original training set (ISIC 2019 + HAM10000).
# Used to compute LDAM margins: Δ_y = C / n_y^{1/4}
_TRAIN_CLASS_COUNTS = {
    "AK":   867,
    "BCC":  3323,
    "BKL":  2624,
    "DF":   239,
    "MEL":  4522,
    "NV":   12875,
    "SCC":  628,
    "UNK":  0,
    "VASC": 253,
}


def fetch_isic_samples(code: str, n: int, skip: int) -> list[dict]:
    import requests as req
    samples: list[dict] = []
    cursor = None
    for field, value in CLASS_QUERIES[code]:
        remaining = n + skip
        while len(samples) < remaining:
            params: dict = {"query": f'{field}:"{value}"', "limit": min(50, remaining)}
            if cursor:
                params["cursor"] = cursor
            r = req.get(ISIC_SEARCH, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            samples.extend(data["results"])
            cursor = data.get("next")
            if cursor:
                cursor = cursor.split("cursor=")[-1].split("&")[0]
            if not cursor or not data["results"]:
                break
    return samples[skip: skip + n]


def download_image(url: str) -> bytes | None:
    import requests as req
    try:
        return req.get(url, timeout=30).content
    except Exception:
        return None


@app.function(
    image=image,
    volumes={"/weights": weights_volume},
    gpu="T4",
    timeout=7200,
    memory=24576,   # 24 GB: B4(~4GB TF graph) + 60img×8cls×380²×3×4B(~1.3GB) + optimizer
)
def fine_tune_model(model_file: str, per_class: int = 60, skip: int = 55, epochs: int = 20):
    """Fine-tune a single EfficientNet model with LDAM loss on balanced ISIC data."""
    import gc
    import sys
    import traceback as tb
    import numpy as np
    import requests as req
    import tf_keras as keras
    from PIL import Image as PILImage

    try:
        print(f"\n{'='*60}", flush=True)
        print(f"Fine-tuning {model_file}", flush=True)
        print(f"{'='*60}\n", flush=True)

        # ── Load model ──────────────────────────────────────────────
        model_path = f"/weights/{model_file}"
        model = keras.models.load_model(model_path, compile=False)
        _, H, W, _ = model.input_shape
        print(f"Loaded {model_file} — input {H}×{W}", flush=True)

        # Freeze all layers, then unfreeze top 30 + classifier
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[-30:]:
            layer.trainable = True
        trainable = sum(p.numpy().size for p in model.trainable_weights)
        print(f"Trainable params: {trainable:,}", flush=True)

        # ── Download balanced training data ─────────────────────────
        X_uint8, y = [], []
        class_counts: dict[str, int] = {}
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for code in [c for c in CLASSES if c != "UNK"]:
            print(f"Fetching {code}…", flush=True)
            samples = fetch_isic_samples(code, per_class, skip)
            idx = CODE_TO_IDX[code]
            ok = 0
            for s in samples:
                raw = download_image(s["files"]["full"]["url"])
                if raw is None:
                    continue
                try:
                    img = PILImage.open(io.BytesIO(raw)).convert("RGB").resize((W, H))
                except Exception:
                    continue
                X_uint8.append(np.asarray(img, dtype=np.uint8))
                y.append(idx)
                ok += 1
            class_counts[code] = ok
            print(f"  {code}: {ok} images", flush=True)

        # Convert to float32 once, then free uint8 list
        X = (np.array(X_uint8, dtype=np.float32) / 255.0 - MEAN) / STD
        del X_uint8
        gc.collect()
        y = np.array(y, dtype=np.int32)
        print(f"\nTotal training images: {len(y)}", flush=True)

        # ── LDAM margins ─────────────────────────────────────────────
        C = 0.5 / max(
            1.0 / max(1, _TRAIN_CLASS_COUNTS[c]) ** 0.25
            for c in CLASSES if c != "UNK"
        )
        margins = np.array([
            C / max(1, _TRAIN_CLASS_COUNTS[c]) ** 0.25 if c != "UNK" else 0.0
            for c in CLASSES
        ], dtype=np.float32)
        print(f"LDAM margins: { {c: f'{margins[i]:.3f}' for i, c in enumerate(CLASSES)} }", flush=True)

        def ldam_loss(y_true, y_pred):
            import tensorflow as tf
            y_true_int = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
            batch_margins = tf.gather(tf.constant(margins), y_true_int)
            one_hot = tf.one_hot(y_true_int, depth=len(CLASSES))
            logits_adj = y_pred - one_hot * batch_margins[:, tf.newaxis]
            return tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits_adj)
            )

        # ── Class weights ────────────────────────────────────────────
        freq = np.array([max(1, class_counts.get(c, 1)) for c in CLASSES], dtype=np.float32)
        class_weights_arr = np.minimum(1.0 / freq / (1.0 / freq).sum() * len(CLASSES), 10.0)
        class_weights = {i: float(class_weights_arr[i]) for i in range(len(CLASSES))}

        # ── One-hot labels ───────────────────────────────────────────
        y_onehot = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
        y_onehot[np.arange(len(y)), y] = 1.0

        lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        drw_split = epochs // 2

        # ── Phase 1: standard CE ────────────────────────────────────
        print(f"\nPhase 1: standard CE, {drw_split} epochs", flush=True)
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(X, y_onehot, batch_size=16, epochs=drw_split, class_weight=class_weights, shuffle=True, callbacks=[lr_cb], verbose=1)

        # ── Phase 2: LDAM ────────────────────────────────────────────
        print(f"\nPhase 2: LDAM loss, {epochs - drw_split} epochs", flush=True)
        model.compile(optimizer=keras.optimizers.Adam(5e-5), loss=ldam_loss, metrics=["accuracy"])
        model.fit(X, y_onehot, batch_size=16, epochs=epochs - drw_split, class_weight=class_weights, shuffle=True, callbacks=[lr_cb], verbose=1)

        # ── Save ─────────────────────────────────────────────────────
        out_path = f"/weights/finetuned_{model_file}"
        model.save(out_path)
        weights_volume.commit()
        print(f"\nSaved → {out_path}", flush=True)
        sys.stdout.flush()
        return out_path

    except Exception:
        tb.print_exc()
        sys.stdout.flush()
        raise


@app.local_entrypoint()
def main(
    model: str = "",
    per_class: int = 60,
    skip: int = 55,
    epochs: int = 20,
):
    """
    Run with:
        modal run scripts/fine_tune_modal.py                          # all 3 models
        modal run scripts/fine_tune_modal.py --model efficientnet_b4.h5  # one model
    """
    models = (
        [model] if model
        else ["efficientnet_b4.h5", "efficientnet_b5.h5", "efficientnet_b7.h5"]
    )

    print(f"Fine-tuning: {models}")
    print(f"Per-class: {per_class}, Skip: {skip}, Epochs: {epochs}\n")

    handles = []
    for m in models:
        handle = fine_tune_model.spawn(m, per_class=per_class, skip=skip, epochs=epochs)
        print(f"Spawned {m} → job id: {handle.object_id}")
        handles.append((m, handle))

    print("\nJobs are running on Modal — your terminal can close safely.")
    print("Check progress at: https://modal.com/apps/sh4wn27/main")
    print()
    print("When complete, run:")
    print("  1. Update ENSEMBLE_FILES in backend/modal_app.py to use finetuned_*.h5")
    print("  2. modal deploy backend/modal_app.py")
    print("  3. python scripts/train_calibrated_head.py --per-class 50 --skip 20")
    print("  4. modal volume put skinai-weights backend/calibration/head.pkl /calibration/head.pkl")
    print("  5. modal deploy backend/modal_app.py")
