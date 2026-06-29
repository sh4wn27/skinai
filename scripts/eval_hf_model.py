"""Evaluate HuggingFace skin lesion classifiers against ISIC held-out data.

Tests two candidates:
  - ALM-AHME/convnextv2-large (HAM10000, 7 classes — best HAM10000 model by downloads)
  - actavkid/vit-large-patch32-384 (12 classes including SCC)

Maps each model's labels onto our 9 CLASSES and reports per-class recall.

Usage:
    python scripts/eval_hf_model.py
    python scripts/eval_hf_model.py --per-class 15 --skip 0
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image as PILImage
from sklearn.metrics import balanced_accuracy_score
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference import CLASSES  # noqa: E402

ISIC_SEARCH = "https://api.isic-archive.com/api/v2/images/search/"
_API_PAGE = 100

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
CODES = list(CLASS_QUERIES.keys())

# Label → CLASSES index for each model
LABEL_MAPS = {
    "ALM-AHME/convnextv2-large-1k-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20": {
        "akiec": 0,  # AK
        "bcc":   1,  # BCC
        "bkl":   2,  # BKL
        "df":    3,  # DF
        "mel":   4,  # MEL
        "nv":    5,  # NV
        "vasc":  8,  # VASC  (no SCC → stays UNK)
    },
    "actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification": {
        "actinic keratosis":    0,  # AK
        "basal cell carcinoma": 1,  # BCC
        "seborrheic keratosis": 2,  # BKL
        "solar lentigo":        2,  # BKL
        "dermatofibroma":       3,  # DF
        "melanoma":             4,  # MEL
        "melanoma metastasis":  4,  # MEL
        "nevus":                5,  # NV
        "squamous cell carcinoma": 6,  # SCC
        "clear skin":           7,  # UNK
        "random":               7,  # UNK
        "vascular lesion":      8,  # VASC
    },
}


def fetch_samples(code: str, n: int, skip: int = 0) -> list[dict]:
    all_results: list[dict] = []
    for field, value in CLASS_QUERIES[code]:
        offset = 0
        while len(all_results) < n + skip:
            batch_limit = min(_API_PAGE, (n + skip) - len(all_results))
            resp = requests.get(
                ISIC_SEARCH,
                params={"query": f'{field}:"{value}"', "limit": batch_limit, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()["results"]
            if not batch:
                break
            all_results.extend(batch)
            offset += len(batch)
            if len(batch) < batch_limit:
                break
    return all_results[skip: skip + n]


def evaluate_model(model_id: str, label_map: dict, imgs: list, y_true: list) -> tuple[list, float]:
    print(f"\n  Loading {model_id}...", file=sys.stderr)
    clf = pipeline("image-classification", model=model_id, top_k=1)

    y_pred = []
    for img in imgs:
        result = clf(img)[0]
        label = result["label"].lower()
        y_pred.append(label_map.get(label, 7))  # default → UNK

    bal = balanced_accuracy_score(y_true, y_pred)
    return y_pred, bal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--skip",      type=int, default=5)
    args = ap.parse_args()

    print(f"Fetching {args.per_class} images/class (skip={args.skip})...", file=sys.stderr)
    imgs, y_true = [], []
    for code in CODES:
        cls_idx = next(i for i, (c, _) in enumerate(CLASSES) if c == code)
        samples = fetch_samples(code, args.per_class, args.skip)
        print(f"  {code}: {len(samples)}", file=sys.stderr)
        for s in samples:
            try:
                img_bytes = requests.get(s["files"]["full"]["url"], timeout=30).content
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                imgs.append(img)
                y_true.append(cls_idx)
            except Exception as e:
                print(f"    skip {s['isic_id']}: {e}", file=sys.stderr)

    classes_present = sorted(set(y_true))
    print(f"\nTotal eval images: {len(imgs)}", file=sys.stderr)

    results = {}
    for model_id, label_map in LABEL_MAPS.items():
        try:
            y_pred, bal = evaluate_model(model_id, label_map, imgs, y_true)
            results[model_id] = (y_pred, bal)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)

    # Print comparison table
    short = {
        "ALM-AHME/convnextv2-large-1k-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20": "ConvNeXt-L (HAM10000)",
        "actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification": "ViT-L (12-class)",
    }
    print(f"\n{'Model':<28} {'Balanced Acc':>14}")
    print("-" * 44)
    for mid, (_, bal) in results.items():
        print(f"{short.get(mid, mid):<28} {bal:>13.1%}")

    print("\n=== Per-class recall ===")
    header = "  {:6}".format("Class")
    for mid in results:
        header += f"  {short.get(mid, mid):>22}"
    print(header)
    print("  " + "-" * (6 + len(results) * 26))
    for i in classes_present:
        n = sum(1 for t in y_true if t == i)
        row = f"  {CLASSES[i][0]:<6}"
        for mid, (y_pred, _) in results.items():
            recall = sum(1 for p, t in zip(y_pred, y_true) if p == i and t == i) / n
            flag = " ✓" if recall >= 0.80 else ""
            row += f"  {recall:>21.1%}{flag}"
        print(row)


if __name__ == "__main__":
    main()
