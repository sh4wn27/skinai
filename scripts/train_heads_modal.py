"""Retrain both calibration heads (EfficientNet + EVA-02) on Modal T4.

Fetches ISIC images per class for each backbone, trains a calibrated LR
head, evaluates on a fresh held-out set, and writes both artifacts to the
Modal volume so inference.py picks them up on next cold start.

Training set : images skip → skip+per_class  (default: 20–169)
Eval set     : images 220–234 (fresh, no overlap with training or old eval)

Usage:
    modal run scripts/train_heads_modal.py
    modal run scripts/train_heads_modal.py --per-class 100   # faster test run

After the run completes, redeploy to load the new heads:
    modal deploy backend/modal_app.py
"""

from __future__ import annotations

import modal
from pathlib import Path

app = modal.App("skinai-train-heads")

_eval_combined = Path(__file__).parent / "eval_combined.py"
_inference = Path(__file__).parent.parent / "backend" / "inference.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Ensure /backend/ exists before copying inference.py there.
    .run_commands("mkdir -p /backend")
    .pip_install(
        "tensorflow==2.20.0",
        "tf-keras==2.20.0",
        "pillow==11.2.1",
        "numpy==2.1.3",
        "scikit-learn==1.5.2",
        "requests==2.32.3",
        "timm==1.0.15",
        "torch==2.6.0",
        "torchvision==0.21.0",
    )
    .add_local_file(str(_eval_combined), "/eval_combined.py")
    .add_local_file(str(_inference), "/backend/inference.py")
)

weights_volume = modal.Volume.from_name("skinai-weights", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/weights": weights_volume},
    gpu="T4",
    timeout=3600,
    memory=20480,
)
def train_heads(
    per_class: int = 150,
    train_skip: int = 20,
    eval_per_class: int = 15,
    eval_skip: int = 220,
) -> None:
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["SKINAI_WEIGHTS_DIR"] = "/weights"
    # Cache EVA-02 weights in the volume so re-runs skip the HF download.
    env["HF_HOME"] = "/weights/hf_cache"
    # Ensure /backend is on PYTHONPATH so `from inference import` works even if
    # the sys.path.insert in eval_combined.py resolves differently in-container.
    env["PYTHONPATH"] = "/backend:" + env.get("PYTHONPATH", "")
    # Show full TF logs so any model-load errors surface immediately.
    env["TF_CPP_MIN_LOG_LEVEL"] = "0"

    cmd = [
        sys.executable, "/eval_combined.py",
        "--train-per-class", str(per_class),
        "--train-skip",      str(train_skip),
        "--eval-per-class",  str(eval_per_class),
        "--eval-skip",       str(eval_skip),
        "--output-dir",      "/weights/calibration",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)

    # Stream subprocess output in real-time so Modal shows progress as it runs,
    # and capture the last N lines to surface in the exception on failure.
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()

    if proc.returncode != 0:
        tail = "".join(captured[-60:])
        raise RuntimeError(
            f"Head training failed (exit {proc.returncode})\n"
            f"--- last 60 lines ---\n{tail}"
        )

    weights_volume.commit()
    print("Volume committed — head.pkl and eva02_head.pkl updated.", flush=True)


@app.local_entrypoint()
def main(per_class: int = 150) -> None:
    train_heads.remote(per_class=per_class)
