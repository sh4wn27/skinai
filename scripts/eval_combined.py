"""Evaluate EfficientNet + EVA-02 soft ensemble.

Both models produce calibrated 9-dim probability vectors.
Final prediction = average of the two.

Usage:
    python scripts/eval_combined.py
    python scripts/eval_combined.py --train-per-class 50 --train-skip 20 --eval-per-class 12 --eval-skip 70
"""

from __future__ import annotations

import argparse
import io
import pickle
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
    "SCC":  [("diagnosis_3", "Squamous cell carcinoma")],
    "VASC": [
        ("diagnosis_3", "Hemangioma"),
        ("diagnosis_3", "Angiokeratoma"),
        ("diagnosis_3", "Pyogenic granuloma"),
    ],
}

CODES = list(CLASS_QUERIES.keys())
CODE_TO_IDX = {c: i for i, (c, _) in enumerate(CLASSES) if c != "UNK"}
# Map indices back — CLASSES has UNK at index 7, VASC at 8
IDX_TO_CODE = {i: c for c, i in CODE_TO_IDX.items()}


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


def eff_raw_probs(ensemble: SkinAIEnsemble, img_bytes: bytes) -> np.ndarray:
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    per_model = []
    for model in ensemble.models:
        _, h, w, _ = model.input_shape
        arr = _preprocess(img, (h, w))
        per_model.append(model.predict(arr, verbose=0)[0])
    return np.mean(per_model, axis=0)  # 9-dim raw probs


def eva_embed(eva_model, transform, img_bytes: bytes) -> np.ndarray:
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = eva_model(tensor)
    return feat.squeeze(0).numpy()  # 768-dim


def collect(ensemble, eva_model, transform, per_class, skip, label):
    eff_X, eva_X, y = [], [], []
    for code in CODES:
        samples = fetch_samples(code, per_class, skip)
        # Map to CLASSES index (includes UNK slot)
        full_idx = next(i for i, (c, _) in enumerate(CLASSES) if c == code)
        print(f"  [{label}] {code}: {len(samples)}", file=sys.stderr)
        for s in samples:
            try:
                img_bytes = requests.get(s["files"]["full"]["url"], timeout=30).content
                ep = eff_raw_probs(ensemble, img_bytes)
                ev = eva_embed(eva_model, transform, img_bytes)
            except Exception as e:
                print(f"    skip {s['isic_id']}: {e}", file=sys.stderr)
                continue
            eff_X.append(np.log(ep + 1e-10))
            eva_X.append(ev)
            y.append(full_idx)
    return np.array(eff_X), np.array(eva_X), np.array(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=50)
    ap.add_argument("--train-skip",      type=int, default=20)
    ap.add_argument("--eval-per-class",  type=int, default=12)
    ap.add_argument("--eval-skip",       type=int, default=70)
    args = ap.parse_args()

    print("Loading EfficientNet ensemble...", file=sys.stderr)
    ensemble = SkinAIEnsemble()

    print("Loading EVA-02-Base...", file=sys.stderr)
    eva_model = timm.create_model("eva02_base_patch14_448", pretrained=True, num_classes=0)
    eva_model.eval()
    cfg = resolve_data_config({}, model=eva_model)
    transform = create_transform(**cfg)

    print("\nCollecting training features...", file=sys.stderr)
    eff_Xtr, eva_Xtr, y_tr = collect(
        ensemble, eva_model, transform,
        args.train_per_class, args.train_skip, "train"
    )

    print("\nCollecting eval features...", file=sys.stderr)
    eff_Xev, eva_Xev, y_ev = collect(
        ensemble, eva_model, transform,
        args.eval_per_class, args.eval_skip, "eval"
    )

    # --- Train EfficientNet LR head ---
    print("\nFitting EfficientNet LR head...", file=sys.stderr)
    eff_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", C=5.0,
                                   max_iter=2000, solver="lbfgs", random_state=42)),
    ])
    eff_pipe.fit(eff_Xtr, y_tr)

    # --- Train EVA-02 LR head ---
    print("Fitting EVA-02 LR head...", file=sys.stderr)
    eva_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", C=5.0,
                                   max_iter=2000, solver="lbfgs", random_state=42)),
    ])
    eva_pipe.fit(eva_Xtr, y_tr)

    # Save the EVA-02 head for deployment.
    # Store classes_ alongside the pipe so inference.py can align the
    # 8-class EVA-02 output into the full 9-class CLASSES space.
    out_dir = Path(__file__).resolve().parent.parent / "backend" / "calibration"
    eva_head_path = out_dir / "eva02_head.pkl"
    with open(eva_head_path, "wb") as f:
        pickle.dump({"pipe": eva_pipe, "classes": eva_pipe.classes_}, f)
    print(f"Saved EVA-02 head → {eva_head_path}", file=sys.stderr)

    # --- Evaluate each model and the soft ensemble ---
    eff_probs_ev = eff_pipe.predict_proba(eff_Xev)   # (N, n_classes)
    eva_probs_ev = eva_pipe.predict_proba(eva_Xev)   # (N, n_classes)
    avg_probs_ev = (eff_probs_ev + eva_probs_ev) / 2  # soft ensemble

    # argmax gives column index — map back to actual class labels via pipe.classes_
    classes_ = eff_pipe.classes_  # e.g. [0,1,2,3,4,5,6,8] (no UNK=7 in training)
    y_eff = classes_[np.argmax(eff_probs_ev, axis=1)]
    y_eva = classes_[np.argmax(eva_probs_ev, axis=1)]
    y_avg = classes_[np.argmax(avg_probs_ev, axis=1)]

    classes_present = sorted(set(y_ev))
    names = [CLASSES[i][0] for i in classes_present]

    def bal(pred): return balanced_accuracy_score(y_ev, pred)

    print(f"\n{'Model':<30} {'Balanced Acc':>14}")
    print("-" * 46)
    print(f"{'EfficientNet + LR head':<30} {bal(y_eff):>13.1%}")
    print(f"{'EVA-02 + LR head':<30} {bal(y_eva):>13.1%}")
    print(f"{'Soft ensemble (avg)':<30} {bal(y_avg):>13.1%}")
    print()

    print("=== Per-class recall ===")
    header = f"{'Class':<8} {'EfficientNet':>14} {'EVA-02':>8} {'Combined':>10}"
    print(header)
    print("-" * 42)
    for i in classes_present:
        n = (y_ev == i).sum()
        r_eff = ((y_eff == i) & (y_ev == i)).sum() / n
        r_eva = ((y_eva == i) & (y_ev == i)).sum() / n
        r_avg = ((y_avg == i) & (y_ev == i)).sum() / n
        winner = "←" if r_avg >= max(r_eff, r_eva) else ""
        print(f"  {CLASSES[i][0]:<6} {r_eff:>14.1%} {r_eva:>8.1%} {r_avg:>10.1%} {winner}")


if __name__ == "__main__":
    main()
