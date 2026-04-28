"""FastAPI app wrapping the SkinAI ensemble."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import SkinAIEnsemble, predict

app = FastAPI(title="SkinAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://skinai.vercel.app"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+|https://.*\.vercel\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Lazy singleton — loading three EfficientNets takes ~10s, so do it on first /predict
# rather than at import. Keeps /health usable before weights are dropped in.
_ensemble: SkinAIEnsemble | None = None


def get_ensemble() -> SkinAIEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = SkinAIEnsemble()
    return _ensemble


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        ensemble = get_ensemble()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        return predict(ensemble, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")
