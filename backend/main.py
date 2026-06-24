"""FastAPI app wrapping the SkinAI ensemble."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from inference import SkinAIEnsemble, predict

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

app = FastAPI(title="SkinAI", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("10/minute")
async def predict_endpoint(request: Request, file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large (max 20 MB)")

    try:
        ensemble = get_ensemble()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        return predict(ensemble, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")
