"""SkinAI benchmark — pulls labeled samples from the ISIC Archive, runs the
production ensemble, and generates a self-contained HTML report with:
  - Per-class precision / recall / F1 / AUC-ROC
  - Confusion matrix heatmap
  - Per-class ROC curves
  - Confidence calibration (reliability diagram)
  - Overall accuracy, balanced accuracy, macro-F1

Usage:
    python scripts/benchmark.py                     # 20 samples/class
    python scripts/benchmark.py --per-class 50      # larger run (~22 min)
    python scripts/benchmark.py --per-class 50 --out report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference import CLASSES, SkinAIEnsemble, _apply_temperature  # noqa: E402
from inference import _open_image, _tta_views, _preprocess          # noqa: E402

ISIC_SEARCH = "https://api.isic-archive.com/api/v2/images/search/"

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
}

CODE_TO_IDX = {code: i for i, (code, _) in enumerate(CLASSES) if code != "UNK"}
EVAL_CODES = list(CLASS_QUERIES.keys())  # 8 classes (no UNK)
EVAL_NAMES = [name for code, name in CLASSES if code in CLASS_QUERIES]


# ── data collection ──────────────────────────────────────────────────────────

def fetch_samples(code: str, n: int) -> list[dict]:
    samples: list[dict] = []
    for field, value in CLASS_QUERIES[code]:
        remaining = n - len(samples)
        if remaining <= 0:
            break
        resp = requests.get(
            ISIC_SEARCH,
            params={"query": f'{field}:"{value}"', "limit": remaining},
            timeout=30,
        )
        resp.raise_for_status()
        samples.extend(resp.json()["results"])
    return samples[:n]


def download_image(url: str) -> bytes:
    return requests.get(url, timeout=30).content


# ── inference ─────────────────────────────────────────────────────────────────

def predict_probs_all_classes(ensemble: SkinAIEnsemble, img_bytes: bytes) -> np.ndarray:
    """Return calibrated probability vector (len=9, matches CLASSES order)."""
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    per_model = []
    for model in ensemble.models:
        _, h, w, _ = model.input_shape
        per_view = [
            model.predict(_preprocess(v, (h, w)), verbose=0)[0]
            for v in _tta_views(img)
        ]
        per_model.append(np.mean(per_view, axis=0))
    raw = np.mean(per_model, axis=0)
    return _apply_temperature(raw)


# ── plotting helpers ──────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> str:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (counts)", fontsize=12, fontweight="bold")
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_roc_curves(y_true_bin: np.ndarray, y_score: np.ndarray, labels: list[str]) -> str:
    n = len(labels)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = axes.flatten()
    for i, (label, ax) in enumerate(zip(labels, axes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=7); ax.set_ylabel("TPR", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("ROC Curves (one-vs-rest)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_per_class_bars(codes: list[str], precision: list[float],
                        recall: list[float], f1: list[float]) -> str:
    x = np.arange(len(codes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w, precision, w, label="Precision", color="#4C72B0")
    ax.bar(x,     recall,    w, label="Recall",    color="#DD8452")
    ax.bar(x + w, f1,        w, label="F1",        color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(codes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax.legend()
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--")
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_calibration(y_true_bin: np.ndarray, y_score: np.ndarray,
                     labels: list[str]) -> str:
    n_bins = 10
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, label in enumerate(labels):
        scores = y_score[:, i]
        true_i = y_true_bin[:, i]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        frac_pos, mean_pred = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() < 3:
                continue
            frac_pos.append(true_i[mask].mean())
            mean_pred.append(scores[mask].mean())
        if mean_pred:
            ax.plot(mean_pred, frac_pos, marker="o", markersize=3, lw=1.2, label=label, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (Reliability Diagram)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML report ───────────────────────────────────────────────────────────────

def build_html(report: dict) -> str:
    ts = report["timestamp"]
    n_total = report["n_total"]
    acc = report["accuracy"]
    bal_acc = report["balanced_accuracy"]
    macro_f1 = report["macro_f1"]
    per_class = report["per_class"]

    rows = ""
    for r in per_class:
        risk = "⚠ HIGH RISK" if r["high_risk"] else ""
        rows += f"""
        <tr>
          <td><b>{r['code']}</b></td>
          <td>{r['name']}</td>
          <td class="num">{r['precision']:.1%}</td>
          <td class="num">{r['recall']:.1%}</td>
          <td class="num">{r['f1']:.1%}</td>
          <td class="num">{r['auc']:.3f}</td>
          <td class="num">{r['support']}</td>
          <td class="risk">{risk}</td>
        </tr>"""

    def img_tag(b64): return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;margin:16px 0">'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>SkinAI Benchmark Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 32px; }}
  .cards {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .card {{ background: #f5f5f5; border-radius: 10px; padding: 20px 28px; min-width: 160px; }}
  .card .val {{ font-size: 2rem; font-weight: 700; color: #333; }}
  .card .lbl {{ font-size: 0.8rem; color: #888; margin-top: 2px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-bottom: 40px; }}
  th {{ background: #f0f0f0; padding: 10px 12px; text-align: left; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eee; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.risk {{ color: #c0392b; font-weight: 600; font-size: 0.8rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 40px; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
  .disclaimer {{ background: #fff8e1; border-left: 4px solid #f9a825; padding: 12px 16px; border-radius: 4px; font-size: 0.85rem; color: #555; margin-top: 40px; }}
</style>
</head>
<body>
<h1>SkinAI Benchmark Report</h1>
<div class="meta">Generated {ts} &nbsp;·&nbsp; {n_total} images across {len(per_class)} classes &nbsp;·&nbsp; ISIC Archive labels</div>

<div class="cards">
  <div class="card"><div class="val">{acc:.1%}</div><div class="lbl">Overall Accuracy</div></div>
  <div class="card"><div class="val">{bal_acc:.1%}</div><div class="lbl">Balanced Accuracy</div></div>
  <div class="card"><div class="val">{macro_f1:.1%}</div><div class="lbl">Macro F1</div></div>
  <div class="card"><div class="val">{n_total}</div><div class="lbl">Images Evaluated</div></div>
</div>

<h2>Per-Class Metrics</h2>
<table>
  <thead><tr><th>Code</th><th>Condition</th><th style="text-align:right">Precision</th><th style="text-align:right">Recall</th><th style="text-align:right">F1</th><th style="text-align:right">AUC</th><th style="text-align:right">n</th><th>Risk</th></tr></thead>
  <tbody>{rows}</tbody>
</table>

<h2>Per-Class Performance</h2>
{img_tag(report['bar_chart'])}

<h2>Confusion Matrix</h2>
{img_tag(report['confusion_matrix'])}

<h2>ROC Curves (one-vs-rest)</h2>
{img_tag(report['roc_curves'])}

<h2>Confidence Calibration</h2>
{img_tag(report['calibration'])}

<div class="disclaimer">
  <b>Research use only.</b> This benchmark uses ISIC Archive images with clinical ground-truth labels.
  Metrics reflect model performance on this sample and are not a substitute for clinical validation.
  SkinAI is a screening tool — not a diagnostic device.
</div>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--out", type=str, default="skinai_benchmark.html")
    args = ap.parse_args()

    print("Loading ensemble...", file=sys.stderr)
    ensemble = SkinAIEnsemble()

    y_true, y_pred, y_score_rows = [], [], []
    skipped = 0

    for true_code in EVAL_CODES:
        samples = fetch_samples(true_code, args.per_class)
        print(f"{true_code}: {len(samples)} samples", file=sys.stderr)
        true_idx = CODE_TO_IDX[true_code]

        for s in samples:
            try:
                img_bytes = download_image(s["files"]["full"]["url"])
                probs = predict_probs_all_classes(ensemble, img_bytes)
            except Exception as e:
                print(f"  skip {s['isic_id']}: {e}", file=sys.stderr)
                skipped += 1
                continue

            pred_idx = int(np.argmax(probs))
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_score_rows.append(probs)
            print(f"  {s['isic_id']}: true={true_code} pred={CLASSES[pred_idx][0]}", file=sys.stderr)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score_rows)

    # filter to eval classes only (drop UNK index=7 from score matrix for AUC)
    eval_indices = [CODE_TO_IDX[c] for c in EVAL_CODES]
    y_score_eval = y_score[:, eval_indices]
    y_true_bin = label_binarize(y_true, classes=eval_indices)

    cr = classification_report(y_true, y_pred, labels=eval_indices,
                               target_names=EVAL_CODES, output_dict=True, zero_division=0)

    per_class = []
    from inference import HIGH_RISK_CODES
    for code in EVAL_CODES:
        idx = eval_indices.index(CODE_TO_IDX[code])
        auc = roc_auc_score(y_true_bin[:, idx], y_score_eval[:, idx]) if y_true_bin[:, idx].sum() > 0 else float("nan")
        per_class.append({
            "code": code,
            "name": dict(CLASSES)[code],
            "precision": cr[code]["precision"],
            "recall": cr[code]["recall"],
            "f1": cr[code]["f1-score"],
            "auc": auc,
            "support": int(cr[code]["support"]),
            "high_risk": code in HIGH_RISK_CODES,
        })

    acc = (y_true == y_pred).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = cr["macro avg"]["f1-score"]

    print("\n=== Summary ===", file=sys.stderr)
    print(f"Overall accuracy:   {acc:.1%}", file=sys.stderr)
    print(f"Balanced accuracy:  {bal_acc:.1%}", file=sys.stderr)
    print(f"Macro F1:           {macro_f1:.1%}", file=sys.stderr)
    print(f"Skipped:            {skipped}", file=sys.stderr)

    print("\nGenerating charts...", file=sys.stderr)
    codes_short = [r["code"] for r in per_class]
    cm = confusion_matrix(y_true, y_pred, labels=eval_indices)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "n_total": len(y_true),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": plot_confusion_matrix(cm, codes_short),
        "roc_curves": plot_roc_curves(y_true_bin, y_score_eval, codes_short),
        "bar_chart": plot_per_class_bars(
            codes_short,
            [r["precision"] for r in per_class],
            [r["recall"] for r in per_class],
            [r["f1"] for r in per_class],
        ),
        "calibration": plot_calibration(y_true_bin, y_score_eval, codes_short),
    }

    out_path = Path(args.out)
    out_path.write_text(build_html(report), encoding="utf-8")
    print(f"\nReport saved → {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
