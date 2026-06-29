"""Train a joint head on concatenated EfficientNet + EVA-02 features.

Architecture:
  EfficientNet log_probs (9-dim) + EVA-02 PCA features (50-dim) → 59-dim
  → GradientBoostingClassifier (class_weight via sample_weight)

This is stronger than averaging two separate heads: one classifier sees both
models' signals and can learn cross-model patterns per class.

Usage:
    python scripts/train_joint_head.py
    python scripts/train_joint_head.py --per-class 150 --skip 20 --eval-skip 200

Outputs:
    backend/calibration/joint_head.pkl   (scaler + PCA + GBM)
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
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
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
N_CLASSES = len(CLASSES)  # 9 (includes UNK)


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


def collect(ensemble: SkinAIEnsemble, eva_model, transform, per_class: int, skip: int, label: str):
    eff_feats, eva_feats, labels = [], [], []
    for code in CODES:
        samples = fetch_samples(code, per_class, skip)
        cls_idx = next(i for i, (c, _) in enumerate(CLASSES) if c == code)
        print(f"  [{label}] {code}: {len(samples)}", file=sys.stderr)
        for s in samples:
            try:
                img_bytes = requests.get(s["files"]["full"]["url"], timeout=30).content
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                # EfficientNet raw log-probs (9-dim)
                per_model = []
                for model in ensemble.models:
                    _, h, w, _ = model.input_shape
                    arr = _preprocess(img, (h, w))
                    per_model.append(model.predict(arr, verbose=0)[0])
                raw = np.mean(per_model, axis=0)
                eff_feat = np.log(raw + 1e-10)
                # EVA-02 embedding (768-dim)
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    eva_feat = eva_model(tensor).squeeze(0).numpy()
            except Exception as e:
                print(f"    skip {s['isic_id']}: {e}", file=sys.stderr)
                continue
            eff_feats.append(eff_feat)
            eva_feats.append(eva_feat)
            labels.append(cls_idx)
    return np.array(eff_feats), np.array(eva_feats), np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class",  type=int, default=150,
                    help="Training images per class")
    ap.add_argument("--skip",       type=int, default=20,
                    help="Skip first N API results (avoids eval-set overlap)")
    ap.add_argument("--eval-per-class", type=int, default=15)
    ap.add_argument("--eval-skip",  type=int, default=200,
                    help="Held-out eval start (skip past training images)")
    ap.add_argument("--pca-dims",   type=int, default=50,
                    help="EVA-02 PCA dimensionality before concatenating")
    args = ap.parse_args()

    print("Loading EfficientNet ensemble...", file=sys.stderr)
    ensemble = SkinAIEnsemble()

    print("Loading EVA-02-Base...", file=sys.stderr)
    eva_model = timm.create_model("eva02_base_patch14_448", pretrained=True, num_classes=0)
    eva_model.eval()
    cfg = resolve_data_config({}, model=eva_model)
    transform = create_transform(**cfg)

    print(f"\nCollecting training features ({args.per_class}/class, skip={args.skip})...",
          file=sys.stderr)
    eff_tr, eva_tr, y_tr = collect(ensemble, eva_model, transform,
                                    args.per_class, args.skip, "train")
    print(f"  Total: {len(y_tr)}", file=sys.stderr)

    print(f"\nCollecting eval features ({args.eval_per_class}/class, skip={args.eval_skip})...",
          file=sys.stderr)
    eff_ev, eva_ev, y_ev = collect(ensemble, eva_model, transform,
                                    args.eval_per_class, args.eval_skip, "eval")
    print(f"  Total: {len(y_ev)}", file=sys.stderr)

    # PCA on EVA-02 features (fit on train only)
    print(f"\nFitting PCA ({args.pca_dims} dims on EVA-02 768-dim)...", file=sys.stderr)
    scaler_eva = StandardScaler()
    pca = PCA(n_components=args.pca_dims, random_state=42)
    eva_tr_scaled = scaler_eva.fit_transform(eva_tr)
    eva_tr_pca = pca.fit_transform(eva_tr_scaled)
    eva_ev_pca = pca.transform(scaler_eva.transform(eva_ev))
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}", file=sys.stderr)

    # Concatenate: EfficientNet log_probs (9) + EVA-02 PCA (50) = 59 dims
    X_tr = np.concatenate([eff_tr, eva_tr_pca], axis=1)
    X_ev = np.concatenate([eff_ev, eva_ev_pca], axis=1)
    print(f"  Joint feature dim: {X_tr.shape[1]}", file=sys.stderr)

    # Gradient boosting with balanced sample weights
    print("\nFitting GradientBoostingClassifier...", file=sys.stderr)
    sw = compute_sample_weight("balanced", y_tr)
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gbm.fit(X_tr, y_tr, sample_weight=sw)

    # Also train a comparison LR head on the same 59-dim features
    from sklearn.linear_model import LogisticRegression
    lr = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", C=5.0,
                                   max_iter=2000, solver="lbfgs", random_state=42)),
    ])
    lr.fit(X_tr, y_tr)

    # Eval
    classes_present = sorted(set(y_ev))

    y_gbm = gbm.predict(X_ev)
    y_lr  = lr.predict(X_ev)
    bal_gbm = balanced_accuracy_score(y_ev, y_gbm)
    bal_lr  = balanced_accuracy_score(y_ev, y_lr)

    print(f"\n{'Model':<35} {'Balanced Acc':>14}")
    print("-" * 51)
    print(f"{'Joint GBM (59-dim)':<35} {bal_gbm:>13.1%}")
    print(f"{'Joint LR (59-dim)':<35} {bal_lr:>13.1%}")
    print()

    print("=== Per-class recall ===")
    print(f"  {'Class':<6}  {'Joint GBM':>10}  {'Joint LR':>10}")
    print("  " + "-" * 30)
    for i in classes_present:
        n = (y_ev == i).sum()
        r_gbm = ((y_gbm == i) & (y_ev == i)).sum() / n
        r_lr  = ((y_lr  == i) & (y_ev == i)).sum() / n
        flag = " ✓" if r_gbm >= 0.80 else ""
        print(f"  {CLASSES[i][0]:<6}  {r_gbm:>10.1%}  {r_lr:>10.1%}{flag}")

    # Save the best model
    best = gbm if bal_gbm >= bal_lr else lr.named_steps["lr"]
    out_path = Path(__file__).resolve().parent.parent / "backend" / "calibration" / "joint_head.pkl"
    payload = {
        "model": gbm,
        "scaler_eva": scaler_eva,
        "pca": pca,
        "classes": np.array([i for i, (c, _) in enumerate(CLASSES) if c != "UNK"]),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved joint head → {out_path}", file=sys.stderr)
    print(f"Use with inference.py: set HAS_JOINT_HEAD=1 or place file at backend/calibration/joint_head.pkl")


if __name__ == "__main__":
    main()
