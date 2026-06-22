"""Fine-tune EfficientNet ensemble on Modal T4 via subprocess.

The training logic lives in fine_tune_standalone.py, which is baked into the
Docker image. The Modal function just calls it via subprocess — no cloudpickle
serialization of training code, no module-level variable capture issues.

Usage:
    modal run scripts/fine_tune_modal.py                           # all 3 models
    modal run scripts/fine_tune_modal.py --model efficientnet_b4.h5
"""

from __future__ import annotations
import modal
from pathlib import Path

app = modal.App("skinai-finetune")

_standalone = Path(__file__).parent / "fine_tune_standalone.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow==2.20.0",
        "tf-keras==2.20.0",
        "pillow==11.2.1",
        "numpy==2.1.3",
        "requests==2.32.3",
    )
    .add_local_file(str(_standalone), "/train.py")
)

weights_volume = modal.Volume.from_name("skinai-weights", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/weights": weights_volume},
    gpu="T4",
    timeout=7200,
    memory=24576,
)
def fine_tune_model(model_file: str, per_class: int = 60, skip: int = 55, epochs: int = 20):
    import subprocess, sys
    cmd = [
        sys.executable, "/train.py",
        model_file,
        "--weights-dir", "/weights",
        "--per-class", str(per_class),
        "--skip", str(skip),
        "--epochs", str(epochs),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training script exited with code {result.returncode}")
    weights_volume.commit()
    print(f"Committed volume — finetuned_{model_file} ready.", flush=True)


@app.local_entrypoint()
def main(
    model: str = "",
    per_class: int = 60,
    skip: int = 55,
    epochs: int = 20,
):
    models = (
        [model] if model
        else ["efficientnet_b4.h5", "efficientnet_b5.h5", "efficientnet_b7.h5"]
    )
    print(f"Fine-tuning: {models}")
    print(f"per_class={per_class}, skip={skip}, epochs={epochs}\n")

    handles = []
    for m in models:
        handle = fine_tune_model.spawn(m, per_class=per_class, skip=skip, epochs=epochs)
        print(f"Spawned {m} → {handle.object_id}")
        handles.append((m, handle))

    print(f"\nWaiting for {len(handles)} job(s) — this process must stay open.")
    print("Logs: https://modal.com/apps/sh4wn27/main\n")

    for m, handle in handles:
        try:
            result = handle.get(timeout=7200)
            print(f"✓ {m} complete → {result}")
        except Exception as e:
            print(f"✗ {m} failed: {e}")

    print("\nAll done. Next steps:")
    print("  1. Update ENSEMBLE_FILES in backend/modal_app.py to finetuned_*.h5")
    print("  2. modal deploy backend/modal_app.py")
    print("  3. python scripts/train_calibrated_head.py --per-class 50 --skip 20")
    print("  4. modal volume put skinai-weights backend/calibration/head.pkl /calibration/head.pkl")
    print("  5. modal deploy backend/modal_app.py")
