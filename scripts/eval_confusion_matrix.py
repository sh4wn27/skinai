"""Pull a small labeled sample from the ISIC Archive per class, run it through
the deployed ensemble (backend/inference.py), and report a confusion matrix +
per-class accuracy. Used to diagnose where the ~80% identification rate is lost.

Usage:
    python scripts/eval_confusion_matrix.py [--per-class N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference import CLASSES, SkinAIEnsemble, predict  # noqa: E402

ISIC_SEARCH = "https://api.isic-archive.com/api/v2/images/search/"

# Each class maps to one or more ISIC diagnosis_2/diagnosis_3 queries (ANDed
# within a query, ORed across the list) discovered by probing the live API.
CLASS_QUERIES: dict[str, list[tuple[str, str]]] = {
    "MEL":  [("diagnosis_2", "Malignant melanocytic proliferations (Melanoma)")],
    "NV":   [("diagnosis_3", "Nevus")],
    "BCC":  [("diagnosis_3", "Basal cell carcinoma")],
    "SCC":  [("diagnosis_2", "Malignant epidermal proliferations")],
    "AK":   [("diagnosis_3", "Solar or actinic keratosis")],
    "BKL":  [("diagnosis_2", "Benign epidermal proliferations")],
    "DF":   [("diagnosis_3", "Dermatofibroma")],
    "VASC": [
        ("diagnosis_3", "Hemangioma"),
        ("diagnosis_3", "Angiokeratoma"),
        ("diagnosis_3", "Pyogenic granuloma"),
        ("diagnosis_3", "Lymphangioma"),
    ],
    # UNK has no real diagnostic taxonomy entry — it's a model catch-all, skip.
}


def fetch_samples(code: str, n: int, skip: int = 0) -> list[dict]:
    """Fetch n samples for a class, skipping the first `skip` results."""
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
    return samples[skip : skip + n]


def download_image(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip first N results per class (use to avoid overlap with training set)")
    args = ap.parse_args()

    print("Loading ensemble (B4 + B5 + B7)...", file=sys.stderr)
    ensemble = SkinAIEnsemble()

    codes = list(CLASS_QUERIES.keys())
    confusion: dict[str, dict[str, int]] = {t: {p: 0 for p, _ in CLASSES} for t in codes}
    total = correct = 0

    for true_code in codes:
        samples = fetch_samples(true_code, args.per_class, skip=args.skip)
        print(f"\n{true_code}: fetched {len(samples)} samples", file=sys.stderr)
        for s in samples:
            try:
                img_bytes = download_image(s["files"]["full"]["url"])
                result = predict(ensemble, img_bytes)
            except Exception as e:  # noqa: BLE001 — eval script, keep going on bad samples
                print(f"  skip {s['isic_id']}: {e}", file=sys.stderr)
                continue
            pred_code = result["predictions"][0]["code"]
            confusion[true_code][pred_code] += 1
            total += 1
            correct += int(pred_code == true_code)
            print(f"  {s['isic_id']}: true={true_code} pred={pred_code} "
                  f"conf={result['predictions'][0]['confidence']:.2f}", file=sys.stderr)

    print("\n=== Confusion matrix (rows=true, cols=predicted) ===")
    header = "true\\pred".ljust(10) + "".join(c.ljust(7) for c, _ in CLASSES)
    print(header)
    for true_code in codes:
        row = confusion[true_code]
        n_true = sum(row.values())
        print(true_code.ljust(10) + "".join(str(row[c]).ljust(7) for c, _ in CLASSES)
              + f"   (n={n_true})")

    print("\n=== Per-class recall ===")
    for true_code in codes:
        row = confusion[true_code]
        n_true = sum(row.values())
        recall = row[true_code] / n_true if n_true else float("nan")
        print(f"{true_code}: {recall:.1%}  (n={n_true})")

    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.1%}" if total else "no samples evaluated")


if __name__ == "__main__":
    main()
