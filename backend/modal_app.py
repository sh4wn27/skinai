"""Modal deployment for SkinAI backend.

Deploy:
    modal deploy modal_app.py

First-time weight upload (run once):
    modal volume create skinai-weights
    modal volume put skinai-weights weights/efficientnet_b4.h5 /efficientnet_b4.h5
    modal volume put skinai-weights weights/efficientnet_b5.h5 /efficientnet_b5.h5
    modal volume put skinai-weights weights/efficientnet_b7.h5 /efficientnet_b7.h5
"""

from pathlib import Path
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
    )
    .copy_local_file(Path(__file__).parent / "inference.py", "/app/inference.py")
    .copy_local_file(Path(__file__).parent / "main.py", "/app/main.py")
    .workdir("/app")
)

weights_volume = modal.Volume.from_name("skinai-weights", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/weights": weights_volume},
    timeout=300,
    container_idle_timeout=300,  # keep warm for 5 min after last request
    cpu=4.0,
    memory=16384,  # 16 GB — B4+B5+B7 together need ~12 GB RAM
)
@modal.asgi_app()
def api():
    import os
    os.environ["SKINAI_WEIGHTS_DIR"] = "/weights"
    from main import app as fastapi_app
    return fastapi_app
