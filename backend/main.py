"""FastAPI app wrapping the SkinAI ensemble."""

from __future__ import annotations

import json

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from dermatologists import PlacesConfigError, geocode_address, nearby_dermatologists
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
async def predict_endpoint(
    request: Request, file: UploadFile = File(...), symptoms: str | None = Form(None)
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large (max 20 MB)")

    parsed_symptoms = None
    if symptoms:
        try:
            parsed_symptoms = json.loads(symptoms)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="symptoms must be valid JSON")

    try:
        ensemble = get_ensemble()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        return predict(ensemble, data, symptoms=parsed_symptoms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")


@app.get("/api/geocode")
@limiter.limit("20/minute")
async def geocode_endpoint(request: Request, address: str) -> dict:
    if not address or not address.strip():
        raise HTTPException(status_code=400, detail="address is required")
    try:
        return await geocode_address(address)
    except PlacesConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"geocoding service error: {e}")


@app.get("/api/dermatologists/nearby")
@limiter.limit("20/minute")
async def nearby_dermatologists_endpoint(
    request: Request, lat: float, lng: float, radius_km: float = 25.0
) -> dict:
    try:
        results = await nearby_dermatologists(lat, lng, radius_km)
    except PlacesConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"places service error: {e}")
    return {"results": results}
