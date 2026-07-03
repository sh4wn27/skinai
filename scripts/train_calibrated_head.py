"""Train a calibrated classification head on top of the frozen EfficientNet ensemble.

Approach: use log(raw_ensemble_probs) as 9-dim features → LogisticRegression with
balanced class weights. This is a strict generalisation of per-class temperature
scaling: temperature scaling learns one scalar per class; LR learns a full 9×9
weight matrix, capturing cross-class confusion patterns.

Usage:
    python scripts/train_calibrated_head.py
    python scripts/train_calibrated_head.py --per-class 40 --skip 20

Outputs:
    backend/calibration/head.pkl   (sklearn pipeline: scaler + LR)
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference import CLASSES, SkinAIEnsemble, _preprocess  # noqa: E402

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
CODES = list(CLASS_QUERIES.keys())
CODE_TO_IDX = {c: i for i, (c, _) in enumerate(CLASSES)}


def fetch_samples(code: str, n: int, skip: int) -> list[dict]:
    """Fetch n samples, skipping the first `skip` (avoids overlap with eval set)."""
    samples: list[dict] = []
    cursor = None

    for field, value in CLASS_QUERIES[code]:
        remaining = n + skip
        while len(samples) < remaining:
            params: dict = {"query": f'{field}:"{value}"', "limit": min(50, remaining)}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(ISIC_SEARCH, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            samples.extend(data["results"])
            cursor = data.get("next")
            if cursor:
                cursor = cursor.split("cursor=")[-1].split("&")[0]
            if not cursor or not data["results"]:
                break

    return samples[skip: skip + n]


def raw_probs(ensemble: SkinAIEnsemble, img_bytes: bytes) -> np.ndarray:
    """Return raw (pre-temperature) 9-dim ensemble probs using single-view inference."""
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    per_model = []
    for model in ensemble.models:
        _, h, w, _ = model.input_shape
        arr = _preprocess(img, (h, w))
        per_model.append(model.predict(arr, verbose=0)[0])
    return np.mean(per_model, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150,
                    help="Training samples per class")
    ap.add_argument("--skip", type=int, default=20,
                    help="Skip first N API results to avoid eval-set overlap")
    ap.add_argument("--val-split", type=float, default=0.25,
                    help="Fraction held out for validation")
    ap.add_argument("--output", type=str, default=None,
                    help="Output path for head.pkl (default: backend/calibration/head.pkl)")
    ap.add_argument("--finetuned", action="store_true",
                    help="Use finetuned_efficientnet_b{4,5,7}.h5 instead of originals")
    args = ap.parse_args()

    print("Loading ensemble...", file=sys.stderr)
    if args.finetuned:
        ensemble = SkinAIEnsemble(
            ensemble_files=("finetuned_efficientnet_b4.h5",
                            "finetuned_efficientnet_b5.h5",
                            "finetuned_efficientnet_b7.h5")
        )
        print("  Using finetuned backbone weights", file=sys.stderr)
    else:
        ensemble = SkinAIEnsemble()

    X_all, y_all = [], []

    for code in CODES:
        samples = fetch_samples(code, args.per_class, args.skip)
        true_idx = CODE_TO_IDX[code]
        print(f"{code}: {len(samples)} samples", file=sys.stderr)
        for s in samples:
            try:
                img_bytes = requests.get(s["files"]["full"]["url"], timeout=30).content
                p = raw_probs(ensemble, img_bytes)
            except Exception as e:
                print(f"  skip {s['isic_id']}: {e}", file=sys.stderr)
                continue
            X_all.append(np.log(p + 1e-10))   # log-prob features
            y_all.append(true_idx)
            print(f"  {s['isic_id']}: true={code} raw_top={CLASSES[np.argmax(p)][0]}({p.max():.2f})",
                  file=sys.stderr)

    X = np.array(X_all)
    y = np.array(y_all)

    # Train / val split (stratified by class)
    rng = np.random.default_rng(42)
    val_mask = np.zeros(len(y), dtype=bool)
    for cls_idx in np.unique(y):
        idx = np.where(y == cls_idx)[0]
        n_val = max(1, int(len(idx) * args.val_split))
        chosen = rng.choice(idx, size=n_val, replace=False)
        val_mask[chosen] = True

    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"\nTrain: {len(y_train)} | Val: {len(y_val)}", file=sys.stderr)

    # Fit pipeline: StandardScaler (log-probs have different ranges) + LR
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            C=5.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    # Validation metrics
    y_pred = pipe.predict(X_val)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    print(f"\nValidation balanced accuracy: {bal_acc:.1%}", file=sys.stderr)

    val_class_indices = sorted(set(y_val))
    cr = classification_report(
        y_val, y_pred,
        labels=val_class_indices,
        target_names=[CLASSES[i][0] for i in val_class_indices],
        zero_division=0,
    )
    print(cr, file=sys.stderr)

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "backend" / "calibration"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "head.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nSaved → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
