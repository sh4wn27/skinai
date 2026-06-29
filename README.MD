# SKINAI

AI-powered skin lesion screening across 9 dermatological conditions. Upload a photo, get an instant risk assessment, and connect with verified dermatologists for follow-up.

> **Disclaimer:** For screening purposes only. Not a substitute for professional medical advice.

---

## What it does

- Classifies skin lesions into 9 categories using a dual-backbone ensemble (EfficientNet B4/B5/B7 + EVA-02-Base)
- A joint GradientBoosting classifier fuses both backbones' features for stronger per-class recall
- Flags high-risk results (MEL, BCC, SCC, AK) and surfaces verified dermatologists

## Conditions detected

| Code | Condition | Risk |
|------|-----------|------|
| MEL  | Melanoma | ⚠️ High |
| NV   | Melanocytic nevus | Low |
| BCC  | Basal cell carcinoma | ⚠️ High |
| AK   | Actinic keratosis | ⚠️ High |
| BKL  | Benign keratosis | Low |
| DF   | Dermatofibroma | Low |
| VASC | Vascular lesion | Low |
| SCC  | Squamous cell carcinoma | ⚠️ High |
| UNK  | Unknown | Low |

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js, Tailwind CSS, TypeScript |
| Backend | FastAPI (Python 3.11) |
| ML — EfficientNet | TensorFlow 2.20 / tf-keras (B4, B5, B7) |
| ML — EVA-02 | PyTorch + timm (`eva02_base_patch14_448`) |
| Classifier | scikit-learn GradientBoostingClassifier (joint 59-dim head) |
| Deployment | Vercel (frontend) · Modal serverless GPU T4 (backend) |

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

Calibration heads go in `backend/calibration/`:
- `head.pkl` — EfficientNet LR head (optional but recommended)
- `eva02_head.pkl` — EVA-02 LR head (optional)
- `joint_head.pkl` — Joint GBM head (preferred; trained by `scripts/train_joint_head.py`)

All `.h5` and `.pkl` files are gitignored and must be generated/downloaded separately.

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

**Inference modes** (auto-selected by which calibration files exist in `backend/calibration/`):

1. `joint_head.pkl` present → **Joint GBM**: single classifier on EfficientNet log-probs (9-dim) + EVA-02 PCA features (50-dim) = 59-dim input
2. `head.pkl` + `eva02_head.pkl` → **Soft ensemble**: average of two separate LR heads
3. `head.pkl` only → **EfficientNet LR head** alone
4. No calibration files → temperature scaling fallback

**EfficientNet ensemble**: B4 (380px) + B5 (456px) + B7 (600px), soft-voted with 4-view TTA (original + 3 flips). Trained on ISIC 2019 + ISIC 2020 + HAM10000.

**EVA-02-Base**: `eva02_base_patch14_448` from timm, 86M params, 448px input, ImageNet-21k pretrained. Used as a frozen feature extractor (768-dim embeddings).

**Joint head**: GradientBoostingClassifier (300 trees, max_depth=4) trained on 150 ISIC images/class. PCA reduces EVA-02 embeddings to 50 dims before concatenation.

## Training calibration heads

```bash
# Train joint head (recommended, ~25 min)
python scripts/train_joint_head.py --per-class 150 --skip 20 --eval-skip 200

# Upload joint head to Modal volume
modal volume put skinai-weights backend/calibration/joint_head.pkl /calibration/joint_head.pkl

# Redeploy
modal deploy backend/modal_app.py
```

## Project structure

```
skinai/
├── backend/
│   ├── inference.py          # ensemble loader, joint head, predict()
│   ├── main.py               # FastAPI app (rate-limited)
│   ├── modal_app.py          # Modal serverless deployment
│   ├── requirements.txt
│   ├── calibration/          # .pkl heads (gitignored)
│   └── weights/              # .h5 backbone weights (gitignored)
├── frontend/                 # Next.js app (Vercel)
│   ├── app/
│   ├── components/
│   └── data/dermatologists.json
├── scripts/
│   ├── train_joint_head.py   # trains joint GBM head (recommended)
│   ├── eval_combined.py      # compares EfficientNet vs EVA-02 vs ensemble
│   ├── eval_eva02.py         # standalone EVA-02 evaluation
│   ├── eval_confusion_matrix.py
│   └── train_calibrated_head.py
└── docs/
    ├── benchmark_report.pdf  # performance evaluation
    └── development_journey.pdf
```

## Author

Huanxuan (Shawn) Li · TJHSST · [sh4wn.org](https://sh4wn.org)
