"""SkinAI inference: EfficientNet B4/B5/B7 ensemble + EVA-02-Base.

Three prediction modes (auto-selected by which calibration files exist):
  1. joint_head.pkl  → single GBM on EfficientNet log_probs (9) + EVA-02 PCA (50)
  2. head.pkl + eva02_head.pkl  → soft average of two separate LR heads
  3. head.pkl only  → EfficientNet LR head (or temperature scaling if absent)
"""

from __future__ import annotations

import base64
import io
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

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

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Per-class temperature scaling fallback (used only when LR head is absent).
# T > 1 BOOSTS a class; T < 1 DAMPENS. Order: AK BCC BKL DF MEL NV SCC UNK VASC
_CLASS_TEMPERATURES = np.array(
    [8.0, 1.5, 0.85, 1.0, 1.5, 0.45, 8.0, 1.0, 1.2], dtype=np.float32
)

WEIGHTS_DIR = Path(os.environ.get("SKINAI_WEIGHTS_DIR", str(Path(__file__).parent / "weights")))
ENSEMBLE_FILES = ("efficientnet_b4.h5", "efficientnet_b5.h5", "efficientnet_b7.h5")


def _calibration_path(filename: str) -> Path | None:
    candidates = [
        Path(os.environ.get("SKINAI_WEIGHTS_DIR", "")) / "calibration" / filename,
        Path(__file__).parent / "calibration" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_calibrated_head():
    p = _calibration_path("head.pkl")
    if p:
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _load_eva02_head() -> tuple | None:
    """Return (pipe, classes_array) or None if eva02_head.pkl not found."""
    p = _calibration_path("eva02_head.pkl")
    if p:
        with open(p, "rb") as f:
            data = pickle.load(f)
        return data["pipe"], data["classes"]
    return None


def _load_joint_head() -> dict | None:
    """Return joint-head dict {model, scaler_eva, pca, classes} or None."""
    p = _calibration_path("joint_head.pkl")
    if p:
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


_calibrated_head = _load_calibrated_head()
_eva02_head_data = _load_eva02_head()
_joint_head     = _load_joint_head()


def _load_eva02_model():
    """Lazy-load EVA-02-Base from timm. Returns (model, transform) or (None, None)."""
    needs_eva = _joint_head is not None or _eva02_head_data is not None
    if not needs_eva:
        return None, None
    try:
        import timm
        import torch  # noqa: F401 — needed by timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        model = timm.create_model("eva02_base_patch14_448", pretrained=True, num_classes=0)
        model.eval()
        transform = create_transform(**resolve_data_config({}, model=model))
        return model, transform
    except Exception:
        return None, None


_eva02_model, _eva02_transform = _load_eva02_model()

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
        _, h, w, _ = model.input_shape
        return (h, w)

    def _raw_ensemble_probs(self, source_image: Image.Image) -> np.ndarray:
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
        raw = self._raw_ensemble_probs(source_image)
        return _calibrate_eff(raw)


def _apply_temperature(probs: np.ndarray) -> np.ndarray:
    logits = np.log(probs + 1e-10) / _CLASS_TEMPERATURES
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _calibrate_eff(raw: np.ndarray) -> np.ndarray:
    if _calibrated_head is not None:
        return _calibrated_head.predict_proba(np.log(raw + 1e-10).reshape(1, -1))[0]
    return _apply_temperature(raw)


def _eva02_probs(img: Image.Image) -> np.ndarray | None:
    """Return EVA-02 calibrated probs aligned to the 9-class CLASSES space, or None."""
    if _eva02_model is None or _eva02_head_data is None:
        return None
    import torch
    pipe, classes_ = _eva02_head_data
    tensor = _eva02_transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = _eva02_model(tensor).squeeze(0).numpy()
    raw = pipe.predict_proba(feat.reshape(1, -1))[0]

    # Expand into full 9-class array (EVA-02 head was trained without UNK samples)
    full = np.zeros(len(CLASSES), dtype=np.float64)
    for col, cls_idx in enumerate(classes_):
        full[cls_idx] = raw[col]
    s = full.sum()
    if s > 0:
        full /= s
    return full


def _joint_probs(raw_eff: np.ndarray, img: Image.Image) -> np.ndarray | None:
    """GBM joint-head on EfficientNet log_probs + EVA-02 PCA features → 9-class probs."""
    if _joint_head is None or _eva02_model is None:
        return None
    import torch
    scaler_eva = _joint_head["scaler_eva"]
    pca        = _joint_head["pca"]
    model      = _joint_head["model"]

    eff_feat = np.log(raw_eff + 1e-10).reshape(1, -1)          # (1, 9)
    tensor = _eva02_transform(img).unsqueeze(0)
    with torch.no_grad():
        eva_raw = _eva02_model(tensor).squeeze(0).numpy()
    eva_feat = pca.transform(scaler_eva.transform(eva_raw.reshape(1, -1)))  # (1, 50)
    X = np.concatenate([eff_feat, eva_feat], axis=1)            # (1, 59)

    probs_partial = model.predict_proba(X)[0]
    classes_seen  = model.classes_

    full = np.zeros(len(CLASSES), dtype=np.float64)
    for col, cls_idx in enumerate(classes_seen):
        full[cls_idx] = probs_partial[col]
    s = full.sum()
    if s > 0:
        full /= s
    return full


def _tta_views(img: Image.Image) -> list[Image.Image]:
    return [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.ROTATE_180),
    ]


def _preprocess(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB").resize((size[1], size[0]))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr[None, ...]


def _open_image(source: ImageSource) -> Image.Image:
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
        return Image.open(source)
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    if isinstance(source, str):
        b64 = source.split(",", 1)[-1]
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    raise ValueError(f"unsupported image source type: {type(source).__name__}")


def predict(ensemble: SkinAIEnsemble, source: ImageSource, top_k: int = 3) -> dict:
    img = _open_image(source)
    raw = ensemble._raw_ensemble_probs(img)   # raw EfficientNet ensemble probs

    if _joint_head is not None and _eva02_model is not None:
        # Mode 1: joint GBM sees both models' features simultaneously
        _p = _joint_probs(raw, img)
        probs = _p if _p is not None else _calibrate_eff(raw)
    else:
        eff_probs = _calibrate_eff(raw)
        eva_probs = _eva02_probs(img)
        probs = (eff_probs + eva_probs) / 2 if eva_probs is not None else eff_probs

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
