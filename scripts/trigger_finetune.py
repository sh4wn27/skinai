"""Trigger fine-tuning on the deployed skinai-finetune app.

Since the app is deployed (modal deploy), spawned calls are persistent —
they run on Modal's servers and survive even if this local script exits.

Usage:
    python3 scripts/trigger_finetune.py
    python3 scripts/trigger_finetune.py efficientnet_b5.h5
"""

import sys
import modal

fine_tune_model = modal.Function.from_name("skinai-finetune", "fine_tune_model")

models = sys.argv[1:] if len(sys.argv) > 1 else ["efficientnet_b5.h5", "efficientnet_b7.h5"]

for m in models:
    handle = fine_tune_model.spawn(m, per_class=60, skip=55, epochs=20)
    print(f"Spawned {m} → {handle.object_id}")

print("\nJobs are running on Modal's servers — safe to close your laptop.")
print("Monitor at: https://modal.com/apps/sh4wn27/main/deployed/skinai-finetune")
