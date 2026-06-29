"""Evaluate EVA-02-Base as a feature extractor for skin lesion classification.

Approach:
  1. Load eva02_base_patch14_448 from timm (pretrained ImageNet-21k → IN1k).
  2. Use it as a frozen feature extractor (num_classes=0 → 768-dim embeddings).
  3. Fetch ISIC samples: train on images skip→skip+per_class, eval on skip+per_class onward.
  4. Train a balanced LR head on training features.
  5. Report per-class recall and balanced accuracy on the held-out set.

Compare these numbers against the EfficientNet + LR head baseline (~47% balanced accuracy,
held-out eval with --skip 70 --per-class 12).

Usage:
    python scripts/eval_eva02.py
    python scripts/eval_eva02.py --train-per-class 50 --train-skip 20 --eval-per-class 15 --eval-skip 70
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import requests
import timm
import torch
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

ISIC_SEARCH = "https://api.isic-archive.com/api/v2/images/search/"

CLASSES: list[tuple[str, str]] = [
    ("AK",   "Actinic keratosis"),
    ("BCC",  "Basal cell carcinoma"),
    ("BKL",  "Benign keratosis"),
    ("DF",   "Dermatofibroma"),
    ("MEL",  "Melanoma"),
    ("NV",   "Melanocytic nevus"),
    ("SCC",  "Squamous cell carcinoma"),
    ("VASC", "Vascular lesion"),
]
CODE_TO_IDX = {c: i for i, (c, _) in enumerate(CLASSES)}

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


def fetch_samples(code: str, n: int, skip: int = 0) -> list[dict]:
    samples: list[dict] = []
    for field, value in CLASS_QUERIES[code]:
        remaining = n + skip - len(samples)
        if remaining <= 0:
            break
        resp = requests.get(
            ISIC_SEARCH,
            params={"query": f'{field}:"{value}"', "limit": remaining},
            timeout=30,
        )
        resp.raise_for_status()
        samples.extend(resp.json()["results"])
    return samples[skip: skip + n]


def embed(model: torch.nn.Module, transform, img_bytes: bytes) -> np.ndarray:
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze(0).numpy()


def collect_features(
    model: torch.nn.Module,
    transform,
    per_class: int,
    skip: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for code in CLASS_QUERIES:
        samples = fetch_samples(code, per_class, skip)
        idx = CODE_TO_IDX[code]
        print(f"  [{label}] {code}: {len(samples)} samples", file=sys.stderr)
        for s in samples:
            try:
                img_bytes = requests.get(s["files"]["full"]["url"], timeout=30).content
                feat = embed(model, transform, img_bytes)
            except Exception as e:
                print(f"    skip {s['isic_id']}: {e}", file=sys.stderr)
                continue
            X.append(feat)
            y.append(idx)
    return np.array(X), np.array(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=50)
    ap.add_argument("--train-skip",      type=int, default=20,
                    help="skip first N API results for training (avoids eval-set overlap)")
    ap.add_argument("--eval-per-class",  type=int, default=12)
    ap.add_argument("--eval-skip",       type=int, default=70,
                    help="skip first N API results for eval (held-out set)")
    args = ap.parse_args()

    print("Loading EVA-02-Base (eva02_base_patch14_448, pretrained)...", file=sys.stderr)
    model = timm.create_model("eva02_base_patch14_448", pretrained=True, num_classes=0)
    model.eval()
    config  = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    print(f"  Feature dim: 768 | Input: 448×448", file=sys.stderr)

    print("\nCollecting training features...", file=sys.stderr)
    X_train, y_train = collect_features(
        model, transform,
        per_class=args.train_per_class,
        skip=args.train_skip,
        label="train",
    )
    print(f"  Total: {len(y_train)} samples", file=sys.stderr)

    print("\nCollecting eval features...", file=sys.stderr)
    X_eval, y_eval = collect_features(
        model, transform,
        per_class=args.eval_per_class,
        skip=args.eval_skip,
        label="eval",
    )
    print(f"  Total: {len(y_eval)} samples", file=sys.stderr)

    print("\nFitting LR head...", file=sys.stderr)
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

    y_pred = pipe.predict(X_eval)
    bal_acc = balanced_accuracy_score(y_eval, y_pred)

    print("\n=== EVA-02-Base + LR head (held-out eval) ===")
    print(f"Balanced accuracy: {bal_acc:.1%}")
    print()

    codes_present = sorted(set(y_eval))
    cr = classification_report(
        y_eval, y_pred,
        labels=codes_present,
        target_names=[CLASSES[i][0] for i in codes_present],
        zero_division=0,
    )
    print(cr)

    print("\n=== Per-class recall ===")
    confusion = {i: {j: 0 for j in range(len(CLASSES))} for i in range(len(CLASSES))}
    for yt, yp in zip(y_eval, y_pred):
        confusion[yt][yp] += 1
    for i in codes_present:
        row = confusion[i]
        n = sum(row.values())
        recall = row[i] / n if n else float("nan")
        print(f"  {CLASSES[i][0]}: {recall:.1%}  (n={n})")

    print(f"\n  EfficientNet baseline (held-out, skip=70): ~47% balanced accuracy")
    print(f"  EVA-02 result: {bal_acc:.1%}")
    if bal_acc > 0.50:
        print("  → EVA-02 improves on baseline. Worth integrating.")
    else:
        print("  → EVA-02 does not beat baseline. Stick with EfficientNet + LR head.")


if __name__ == "__main__":
    main()
