"""SkinAI inference: EfficientNet B4/B5/B7 soft-voting ensemble.

Trained on ISIC 2019 + ISIC 2020 + HAM10000 for 9-class skin-lesion classification.
Output JSON shape matches the API spec in the project CLAUDE.md.
"""

from __future__ import annotations

import base64
import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import os

import numpy as np
from PIL import Image
import tf_keras  # Keras 2 compatibility layer — loads .h5 weights saved before Keras 3

# Alphabetical order of diagnosis codes — matches flow_from_dataframe's default sort.
# Verified against the upstream Tirth27 training pipeline (main_run.py class_indices).
CLASSES: list[tuple[str, str]] = [
    ("AK",   "Actinic keratosis"),
    ("BCC",  "Basal cell carcinoma"),
    ("BKL",  "Benign keratosis"),
    ("DF",   "Dermatofibroma"),
    ("MEL",  "Melanoma"),
    ("NV",   "Melanocytic nevus"),
    ("SCC",  "Squamous cell carcinoma"),
    ("UNK",  "Unknown"),
    ("VASC", "Vascular lesion"),
]
HIGH_RISK_CODES = {"MEL", "BCC", "SCC", "AK"}
DISCLAIMER = "For screening purposes only. Not a substitute for professional medical advice."

# ImageNet statistics — matches A.Normalize() used in training (pre_train.py).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Per-class temperature scaling: logit_c = log(p_c) / T_c, then re-softmax.
# Math: log(p) is always negative. Dividing by T > 1 makes it less negative → higher prob.
# So T > 1 BOOSTS a class; T < 1 DAMPENS it. (Previous values were reversed — fixed Jun 2026.)
# Probe of raw model probs on ISIC samples showed:
#   SCC: up to 0.174 raw (rank-2 signal present) → T=4.0 can surface it
#   AK:  up to 0.102 raw (rank-3 signal present) → T=4.0 can surface it
#   DF:  max 0.004 raw (no signal)               → neutral T; needs retraining
#   NV:  0.85–0.97 raw (dominant)                → T=0.5 dampens it
# Order: AK   BCC  BKL  DF   MEL  NV   SCC  UNK  VASC
_CLASS_TEMPERATURES = np.array(
    [8.0, 1.5, 0.85, 1.0, 1.5, 0.45, 8.0, 1.0, 1.2], dtype=np.float32
)  # AK   BCC  BKL   DF   MEL  NV    SCC  UNK  VASC

WEIGHTS_DIR = Path(os.environ.get("SKINAI_WEIGHTS_DIR", str(Path(__file__).parent / "weights")))
ENSEMBLE_FILES = ("efficientnet_b4.h5", "efficientnet_b5.h5", "efficientnet_b7.h5")

# Calibrated head: sklearn Pipeline (StandardScaler + LogisticRegression) trained on
# balanced ISIC samples. Takes log(raw_probs) as input, outputs calibrated class probs.
# Falls back to per-class temperature scaling when not present.
# Checks Modal volume path (/weights/calibration/head.pkl) before local path.
def _load_calibrated_head():
    candidates = [
        Path(os.environ.get("SKINAI_WEIGHTS_DIR", "")) / "calibration" / "head.pkl",
        Path(__file__).parent / "calibration" / "head.pkl",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    return None

_calibrated_head = _load_calibrated_head()

ImageSource = Union[str, bytes, Path]


@dataclass
class Prediction:
    class_name: str
    code: str
    confidence: float


class SkinAIEnsemble:
    """Soft-voting ensemble of EfficientNet B4/B5/B7."""

    def __init__(
        self,
        weights_dir: Path = WEIGHTS_DIR,
        ensemble_files: Sequence[str] = ENSEMBLE_FILES,
    ):
        self.models: list[tf_keras.Model] = []
        for fname in ensemble_files:
            path = weights_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing weights file: {path}. "
                    "Drop the pretrained .h5 files into backend/weights/."
                )
            self.models.append(tf_keras.models.load_model(path, compile=False))

    def input_size(self, model: tf_keras.Model) -> tuple[int, int]:
        """Read the (H, W) this model expects directly from its input tensor."""
        _, h, w, _ = model.input_shape
        return (h, w)

    def _raw_ensemble_probs(self, source_image: Image.Image) -> np.ndarray:
        """TTA soft-vote across all models, returning raw (uncalibrated) probs."""
        per_model: list[np.ndarray] = []
        for model in self.models:
            h, w = self.input_size(model)
            per_view = [
                model.predict(_preprocess(view, (h, w)), verbose=0)[0]
                for view in _tta_views(source_image)
            ]
            per_model.append(np.mean(per_view, axis=0))
        return np.mean(per_model, axis=0)

    def predict_probs(self, source_image: Image.Image) -> np.ndarray:
        """Return calibrated class probabilities.

        Uses the learned LR calibration head when available (trained on balanced
        ISIC data), falling back to per-class temperature scaling.
        """
        raw = self._raw_ensemble_probs(source_image)
        return _calibrate(raw)


def _apply_temperature(probs: np.ndarray) -> np.ndarray:
    """Per-class temperature scaling on log-probability space, then re-softmax."""
    logits = np.log(probs + 1e-10) / _CLASS_TEMPERATURES
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _calibrate(raw: np.ndarray) -> np.ndarray:
    """Apply calibrated LR head if available; otherwise temperature scaling."""
    if _calibrated_head is not None:
        log_probs = np.log(raw + 1e-10).reshape(1, -1)
        return _calibrated_head.predict_proba(log_probs)[0]
    return _apply_temperature(raw)


def _tta_views(img: Image.Image) -> list[Image.Image]:
    """Original + horizontal flip + vertical flip + 180° rotation."""
    return [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.ROTATE_180),
    ]


def _preprocess(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize + ImageNet normalise → (1, H, W, 3) float32."""
    img = img.convert("RGB").resize((size[1], size[0]))  # PIL uses (W, H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr[None, ...]


def _open_image(source: ImageSource) -> Image.Image:
    """Decode an image from a file path, raw bytes, or base64 string."""
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
        return Image.open(source)
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    if isinstance(source, str):
        b64 = source.split(",", 1)[-1]  # tolerate data:image/...;base64, prefix
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    raise ValueError(f"unsupported image source type: {type(source).__name__}")


def predict(ensemble: SkinAIEnsemble, source: ImageSource, top_k: int = 3) -> dict:
    """Run inference and return the JSON response shape used by /predict."""
    img = _open_image(source)
    probs = ensemble.predict_probs(img)
    order = np.argsort(probs)[::-1][:top_k]

    top = [
        Prediction(
            class_name=CLASSES[i][1],
            code=CLASSES[i][0],
            confidence=float(probs[i]),
        )
        for i in order
    ]

    return {
        "predictions": [
            {"class": p.class_name, "code": p.code, "confidence": round(p.confidence, 4)}
            for p in top
        ],
        "high_risk": top[0].code in HIGH_RISK_CODES,
        "disclaimer": DISCLAIMER,
        "top_class": top[0].class_name,
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="SkinAI ensemble inference (CLI).")
    ap.add_argument("image", help="path to image file")
    args = ap.parse_args()

    try:
        ensemble = SkinAIEnsemble()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(predict(ensemble, args.image), indent=2))
