# SkinAI

AI-powered skin lesion screening across 9 dermatological conditions. Upload a photo, get an instant risk assessment, and connect with verified dermatologists for follow-up.

> **Disclaimer:** For screening purposes only. Not a substitute for professional medical advice.

---

## What it does

- Classifies skin lesions into 9 categories using an EfficientNet B4/B5/B7 ensemble
- Flags high-risk results (melanoma, BCC, SCC) and surfaces verified dermatologists
- 20,000+ users · 10+ verified dermatologists · pending patent

## Conditions detected

| Code | Condition |
|------|-----------|
| MEL | Melanoma ⚠️ |
| NV | Melanocytic nevus |
| BCC | Basal cell carcinoma ⚠️ |
| AK | Actinic keratosis |
| BKL | Benign keratosis |
| DF | Dermatofibroma |
| VASC | Vascular lesion |
| SCC | Squamous cell carcinoma ⚠️ |
| UNK | Unknown |

⚠️ = high-risk, triggers dermatologist recommendation

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, Tailwind CSS, TypeScript |
| Backend | FastAPI (Python) |
| ML model | EfficientNet B4/B5/B7 ensemble (TensorFlow/Keras) |
| Deployment | Vercel (frontend) · Modal (backend GPU) |

## Running locally

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Weights go in `backend/weights/` as:
- `efficientnet_b4.h5`
- `efficientnet_b5.h5`
- `efficientnet_b7.h5`

They are gitignored and must be downloaded separately.

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -F "file=@image.jpg" http://localhost:8000/predict
```

Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API

### `POST /predict`

Accepts a multipart image upload. Returns top-3 predictions with confidence scores.

```json
{
  "predictions": [
    { "class": "Melanoma", "code": "MEL", "confidence": 0.72 },
    { "class": "Basal cell carcinoma", "code": "BCC", "confidence": 0.15 },
    { "class": "Melanocytic nevus", "code": "NV", "confidence": 0.08 }
  ],
  "high_risk": true,
  "top_class": "Melanoma",
  "disclaimer": "For screening purposes only. Not a substitute for professional medical advice."
}
```

### `GET /health`

Returns `{ "status": "ok" }`.

## Model

- **Architecture:** EfficientNet B4/B5/B7 soft-voting ensemble
- **Training data:** ISIC 2019 + ISIC 2020 + HAM10000
- **Preprocessing:** ImageNet normalization (`A.Normalize()`)
- **Weights format:** `.h5` (Keras 2), loaded via `tf-keras` compatibility layer

## Project structure

```
skinai/
├── backend/
│   ├── inference.py      # ensemble loader + predict()
│   ├── main.py           # FastAPI app
│   ├── requirements.txt
│   └── weights/          # .h5 files (gitignored)
└── frontend/             # Next.js app (coming)
```

## Author

Huanxuan (Shawn) Li · TJHSST · [sh4wn.org](https://sh4wn.org)

Built on top of [Tirth27/Skin-Cancer-Classification-using-Deep-Learning](https://github.com/Tirth27/Skin-Cancer-Classification-using-Deep-Learning).
