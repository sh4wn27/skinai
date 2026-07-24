"""Google Places/Geocoding integration for the nearest-dermatologist finder.

Pure directory lookup against Google's live data — no curated/onboarded list,
no verification claims. See active/decisions or project notes for rationale.
"""

from __future__ import annotations

import math
import os

import httpx

_PLACES_SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.rating,places.userRatingCount,places.nationalPhoneNumber,places.googleMapsUri"
)


class PlacesConfigError(RuntimeError):
    """Raised when GOOGLE_MAPS_API_KEY is not configured."""


def _api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise PlacesConfigError("GOOGLE_MAPS_API_KEY is not set")
    return key


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


async def geocode_address(address: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(_GEOCODE_URL, params={"address": address, "key": _api_key()})
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK" or not data.get("results"):
        raise ValueError(f"could not geocode address: {data.get('status', 'UNKNOWN')}")
    top = data["results"][0]
    loc = top["geometry"]["location"]
    return {"lat": loc["lat"], "lng": loc["lng"], "formatted_address": top["formatted_address"]}


async def nearby_dermatologists(lat: float, lng: float, radius_km: float = 25.0) -> list[dict]:
    body = {
        "textQuery": "dermatologist",
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius_km * 1000.0,
            }
        },
        "maxResultCount": 20,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _api_key(),
        "X-Goog-FieldMask": _FIELD_MASK,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(_PLACES_SEARCH_TEXT_URL, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for place in data.get("places", []):
        loc = place.get("location", {})
        p_lat, p_lng = loc.get("latitude"), loc.get("longitude")
        distance_km = (
            round(_haversine_km(lat, lng, p_lat, p_lng), 2)
            if p_lat is not None and p_lng is not None
            else None
        )
        results.append(
            {
                "id": place.get("id"),
                "name": place.get("displayName", {}).get("text", "Unknown"),
                "address": place.get("formattedAddress"),
                "lat": p_lat,
                "lng": p_lng,
                "distance_km": distance_km,
                "rating": place.get("rating"),
                "rating_count": place.get("userRatingCount"),
                "phone": place.get("nationalPhoneNumber"),
                "maps_url": place.get("googleMapsUri"),
            }
        )
    results.sort(key=lambda r: r["distance_km"] if r["distance_km"] is not None else float("inf"))
    return results
