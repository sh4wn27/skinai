"""SkinAI inference: EfficientNet B4/B5/B7 soft-voting ensemble.

Trained on ISIC 2019 + ISIC 2020 + HAM10000 for 9-class skin-lesion classification.
Output JSON shape matches the API spec in the project CLAUDE.md.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image
import tensorflow as tf

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
HIGH_RISK_CODES = {"MEL", "BCC", "SCC"}
DISCLAIMER = "For screening purposes only. Not a substitute for professional medical advice."

# ImageNet statistics — matches A.Normalize() used in training (pre_train.py).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WEIGHTS_DIR = Path(__file__).parent / "weights"
ENSEMBLE_FILES = ("efficientnet_b4.h5", "efficientnet_b5.h5", "efficientnet_b7.h5")

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
        self.models: list[tf.keras.Model] = []
        for fname in ensemble_files:
            path = weights_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing weights file: {path}. "
                    "Drop the pretrained .h5 files into backend/weights/."
                )
            self.models.append(tf.keras.models.load_model(path, compile=False))

    def input_size(self, model: tf.keras.Model) -> tuple[int, int]:
        """Read the (H, W) this model expects directly from its input tensor."""
        _, h, w, _ = model.input_shape
        return (h, w)

    def predict_probs(self, source_image: Image.Image) -> np.ndarray:
        """Preprocess per-model, run inference, return soft-voted probabilities."""
        per_model: list[np.ndarray] = []
        for model in self.models:
            h, w = self.input_size(model)
            arr = _preprocess(source_image, (h, w))
            per_model.append(model.predict(arr, verbose=0)[0])
        return np.mean(per_model, axis=0)


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
