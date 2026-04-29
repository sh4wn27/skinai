# CLAUDE.md — SkinAI

This file gives Claude Code full context on the SkinAI project. Read this before touching any code.

---

## What is SkinAI

SkinAI is an AI-powered skin cancer screening web app. Users upload a photo of a skin lesion and get an instant risk assessment across 9 dermatological conditions. High-risk results surface a curated network of verified dermatologists for follow-up.

This is NOT a medical device. Every response and UI element must include a disclaimer: "For screening purposes only. Not a substitute for professional medical advice."

**Status:** Active development. Forked from open-source base, being rebuilt with custom frontend and dermatologist network layer.

---

## Founder context

Built by Shawn (Huanxuan Li), junior at TJHSST (Thomas Jefferson High School for Science and Technology) in Northern Virginia. Shawn has a personal connection to transplant medicine — he received a cord blood transplant as a child for beta-thalassemia. He also runs CAPA (a transplant immunology ML framework), Growing Up with Robotics (a 501c3 nonprofit), and FTC robotics team 14607.

SkinAI has 20,000+ users and a pending patent. The dermatologist network (10+ verified dermatologists onboarded) is the core differentiator — not just the ML model.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, TypeScript |
| Deployment (frontend) | Vercel |
| Backend API | FastAPI (Python) |
| Deployment (backend) | Modal (serverless GPU) |
| ML model | EfficientNet B4/B5/B7 ensemble, TensorFlow/Keras |
| Weights format | .h5 files |
| Email / forms | EmailJS (no backend needed) |
| Auth | None for now |

---

## Repo structure

```
skinai/
├── CLAUDE.md                  ← you are here
├── README.md
├── frontend/                  ← Next.js app
│   ├── app/
│   │   ├── page.tsx           ← landing page
│   │   ├── scan/
│   │   │   └── page.tsx       ← image upload + prediction UI
│   │   └── dermatologists/
│   │       └── page.tsx       ← dermatologist directory
│   ├── components/
│   │   ├── DisclaimerModal.tsx
│   │   ├── ResultCard.tsx
│   │   └── DermCard.tsx
│   ├── data/
│   │   └── dermatologists.json  ← static dermatologist database
│   └── public/
├── backend/                   ← FastAPI + Modal
│   ├── inference.py           ← model loading + prediction logic
│   ├── modal_app.py           ← Modal deployment config
│   ├── main.py                ← FastAPI app
│   └── weights/               ← .h5 model weights (gitignored)
└── model/                     ← training notebooks (reference only, don't touch)
```

---

## ML model details

- **Architecture:** EfficientNet B4, B5, B7 ensemble (soft voting on probabilities)
- **Dataset:** ISIC 2020 + ISIC 2019 + HAM10000
- **Output classes (9):**
  1. Melanoma (MEL)
  2. Melanocytic nevus (NV)
  3. Basal cell carcinoma (BCC)
  4. Actinic keratosis (AK)
  5. Benign keratosis (BKL)
  6. Dermatofibroma (DF)
  7. Vascular lesion (VASC)
  8. Squamous cell carcinoma (SCC)
  9. Unknown / other (UNK)
- **High-risk classes** (trigger dermatologist recommendation): MEL, BCC, SCC
- **Input:** 224x224 RGB image, normalized to [0,1]
- **Weights location:** `backend/weights/` — never commit these to git

---

## API spec

### POST /predict
Accepts a multipart image upload. Returns JSON.

**Request:**
```
Content-Type: multipart/form-data
file: <image file>
```

**Response:**
```json
{
  "predictions": [
    { "class": "Melanoma", "code": "MEL", "confidence": 0.72 },
    { "class": "Basal cell carcinoma", "code": "BCC", "confidence": 0.15 },
    { "class": "Melanocytic nevus", "code": "NV", "confidence": 0.08 }
  ],
  "high_risk": true,
  "disclaimer": "For screening purposes only. Not a substitute for professional medical advice.",
  "top_class": "Melanoma"
}
```

### GET /health
Returns `{ "status": "ok" }`.

**CORS:** Allow `https://skinai.vercel.app` and `http://localhost:3000`.

---

## Frontend behavior

### Landing page (`/`)
- Hero section: headline, subheadline, CTA button → /scan
- Stats row: "20,000+ Users", "10+ Dermatologists", "9 Conditions Detected"
- How it works: 3-step visual (Upload → Analyze → Connect)
- Waitlist form (EmailJS) for dermatologists wanting to join the network
- Footer with disclaimer

### Scan page (`/scan`)
- On first visit: show `DisclaimerModal` — user must click "I understand" to proceed
- Drag-and-drop image upload (also click to browse)
- On submit: POST to Modal API, show loading state
- On result: show `ResultCard` with top-3 predictions + confidence bars
- If `high_risk: true`: show "See a specialist" section below result, surface 2-3 DermCards from `dermatologists.json`

### Dermatologists page (`/dermatologists`)
- Grid of `DermCard` components
- Each card: name, photo, specialty, location, "Request consultation" button (mailto link)
- Filter by specialty (optional, nice to have)

---

## Dermatologist data format

`frontend/data/dermatologists.json`:
```json
[
  {
    "id": "1",
    "name": "Dr. Jane Smith",
    "specialty": "Dermatology",
    "subspecialty": "Melanoma / Skin Cancer",
    "location": "Washington, DC",
    "hospital": "Georgetown University Hospital",
    "email": "jsmith@example.com",
    "photo": "/derms/jsmith.jpg",
    "verified": true
  }
]
```

---

## Design system

- **Theme:** Dark (near-black background, white text)
- **Accent color:** Teal / cyan (#00C9A7 or similar)
- **Font:** Sharp, clean sans-serif — avoid Inter/Roboto
- **Tone:** Clinical but approachable. Not scary, not playful.
- **Always visible:** Medical disclaimer on every page, not buried in footer

---

## Environment variables

```
# backend/.env
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
MODEL_WEIGHTS_PATH=./weights/

# frontend/.env.local
NEXT_PUBLIC_API_URL=https://skinai--predict.modal.run
NEXT_PUBLIC_EMAILJS_SERVICE_ID=...
NEXT_PUBLIC_EMAILJS_TEMPLATE_ID=...
NEXT_PUBLIC_EMAILJS_PUBLIC_KEY=...
```

---

## Hard rules — never violate these

1. **Medical disclaimer on everything.** Every API response, every scan result, landing page footer, disclaimer modal. SkinAI is a screening tool, not a diagnostic device.
2. **Never commit model weights.** `.h5` files go in `backend/weights/` which is gitignored.
3. **No auth for now.** Keep it open. Don't add login flows or user accounts yet.
4. **No dark patterns.** No fake urgency, no manipulative CTAs. Users are people worried about their skin.
5. **Fail gracefully.** If the Modal API is down, show a clear error message, not a crash.
6. **Mobile first.** The primary use case is someone taking a photo on their phone.

---

## Phase reference (where we are in the build)

| Phase | Description | Status |
|---|---|---|
| 1 | Fork base repo, get inference running locally | TODO |
| 2 | FastAPI + Modal deployment | TODO |
| 3 | Next.js frontend (landing, scan, derms pages) | TODO |
| 4 | Dermatologist network layer + EmailJS | TODO |
| 5 | Deploy, polish, publish | TODO |

Update this table as phases complete.

---

## Key commands

```bash
# Backend
cd backend
pip install fastapi uvicorn tensorflow pillow numpy python-multipart
uvicorn main:app --reload

# Deploy to Modal
python3 -m modal deploy modal_app.py

# Frontend
cd frontend
npm install
npm run dev

# Deploy frontend
vercel --prod
```

---

## Contact

Shawn (Huanxuan Li) — sh4wn.org — GitHub: sh4wn27