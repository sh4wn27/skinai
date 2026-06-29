"""Modal deployment for SkinAI backend.

Deploy:
    modal deploy modal_app.py

First-time weight upload (run once):
    modal volume create skinai-weights
    modal volume put skinai-weights weights/efficientnet_b4.h5 /efficientnet_b4.h5
    modal volume put skinai-weights weights/efficientnet_b5.h5 /efficientnet_b5.h5
    modal volume put skinai-weights weights/efficientnet_b7.h5 /efficientnet_b7.h5
"""

import modal

app = modal.App("skinai")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.115.0",
        "python-multipart==0.0.9",
        "tensorflow==2.20.0",
        "tf-keras==2.20.0",
        "pillow==11.2.1",
        "numpy==2.1.3",
        "scikit-learn==1.5.2",  # calibrated LR head
        "slowapi==0.1.9",       # rate limiting
        "timm==1.0.15",         # EVA-02-Base feature extractor
        "torch==2.6.0",         # timm dependency (CPU-only for feature extraction)
        "torchvision==0.21.0",
    )
    .add_local_python_source("inference", "main")
)

weights_volume = modal.Volume.from_name("skinai-weights", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/weights": weights_volume},
    gpu="T4",
    timeout=300,
    scaledown_window=300,  # keep warm for 5 min after last request
    memory=20480,  # 20 GB — EfficientNet ~12 GB + EVA-02 ~2 GB
)
@modal.asgi_app()
def api():
    import os
    os.environ["SKINAI_WEIGHTS_DIR"] = "/weights"
    # Cache timm/HF model downloads in the persistent volume so EVA-02
    # is only downloaded once (first cold start) and reused thereafter.
    os.environ["HF_HOME"] = "/weights/hf_cache"
    from main import app as fastapi_app
    return fastapi_app
