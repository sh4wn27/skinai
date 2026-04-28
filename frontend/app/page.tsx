import Link from 'next/link'
import Nav from '@/components/Nav'
import ScanWidget from '@/components/ScanWidget'

const stats = [
  { value: '20,000+', label: 'Users Screened' },
  { value: '9', label: 'Conditions Detected' },
  { value: '10+', label: 'Verified Dermatologists' },
]

const steps = [
  {
    title: 'Upload',
    desc: 'Take or upload a clear photo of the skin lesion. JPG, PNG, or WEBP work great.',
  },
  {
    title: 'Analyze',
    desc: 'Our EfficientNet ensemble screens for 9 dermatological conditions in seconds.',
  },
  {
    title: 'Connect',
    desc: 'High-risk results surface verified dermatologists for expert follow-up.',
  },
]

const conditions = [
  { code: 'MEL', name: 'Melanoma', highRisk: true },
  { code: 'NV', name: 'Melanocytic nevus', highRisk: false },
  { code: 'BCC', name: 'Basal cell carcinoma', highRisk: true },
  { code: 'AK', name: 'Actinic keratosis', highRisk: false },
  { code: 'BKL', name: 'Benign keratosis', highRisk: false },
  { code: 'DF', name: 'Dermatofibroma', highRisk: false },
  { code: 'VASC', name: 'Vascular lesion', highRisk: false },
  { code: 'SCC', name: 'Squamous cell carcinoma', highRisk: true },
  { code: 'UNK', name: 'Unknown', highRisk: false },
]

export default function HomePage() {
  return (
    <main className="min-h-screen bg-white">
      <Nav />

      {/* Hero */}
      <section className="pt-36 pb-24 px-6 text-center">
        <div className="max-w-4xl mx-auto">
          <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-pink-50 text-pink-800 rounded-full text-sm font-medium mb-8 border border-pink-100">
            <span className="w-2 h-2 rounded-full bg-pink-700 animate-pulse" />
            20,000+ users screened
          </span>
          <h1 className="text-6xl md:text-8xl font-bold tracking-tight text-zinc-900 mb-6 leading-[1.05]">
            AI Skin Cancer
            <br />
            <span className="text-pink-700">Screening.</span>
          </h1>
          <p className="text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Upload a photo of your skin lesion. Get an instant risk assessment across 9
            dermatological conditions. Connect with verified dermatologists.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link
              href="/scan"
              className="px-8 py-4 bg-pink-700 text-white rounded-full font-semibold hover:bg-pink-800 transition-colors text-lg shadow-lg shadow-pink-100"
            >
              Get Screened Free
            </Link>
            <Link
              href="/dermatologists"
              className="px-8 py-4 border-2 border-zinc-200 text-zinc-600 rounded-full font-semibold hover:border-zinc-300 hover:text-zinc-900 transition-colors text-lg"
            >
              Our Dermatologists →
            </Link>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-12 border-y border-zinc-100 bg-zinc-50/50">
        <div className="max-w-3xl mx-auto grid grid-cols-3 gap-8 text-center px-6">
          {stats.map((s) => (
            <div key={s.label}>
              <div className="text-4xl md:text-5xl font-bold text-pink-700">{s.value}</div>
              <div className="text-zinc-400 mt-2 text-sm md:text-base">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Try it live */}
      <section className="py-24 px-6">
        <div className="max-w-2xl mx-auto text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-zinc-900 mb-4">Try it live</h2>
          <p className="text-zinc-400 text-lg">
            Upload a photo and see the AI analyze it in real time.
          </p>
          <p className="text-zinc-300 text-sm mt-2">
            For screening purposes only — not a medical diagnosis.
          </p>
        </div>
        <ScanWidget showDisclaimer={false} />
      </section>

      {/* How it works */}
      <section className="py-24 px-6 bg-zinc-50/50 border-y border-zinc-100">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-zinc-900">How it works</h2>
            <p className="text-zinc-400 mt-4 text-lg">Three steps from photo to clarity.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-12">
            {steps.map((step, i) => (
              <div key={i} className="text-center">
                <div className="w-14 h-14 rounded-2xl bg-white border-2 border-pink-100 flex items-center justify-center mx-auto mb-6 shadow-sm shadow-pink-50">
                  <span className="text-2xl font-bold text-pink-700">{i + 1}</span>
                </div>
                <h3 className="text-xl font-bold text-zinc-900 mb-3">{step.title}</h3>
                <p className="text-zinc-400 leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Conditions */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-zinc-900 mb-3">9 conditions detected</h2>
          <p className="text-zinc-400 mb-10">
            Trained on ISIC 2019 + ISIC 2020 + HAM10000
          </p>
          <div className="flex flex-wrap gap-3 justify-center">
            {conditions.map((c) => (
              <span
                key={c.code}
                className={`px-4 py-2 rounded-full text-sm font-medium border ${
                  c.highRisk
                    ? 'bg-red-50 text-red-600 border-red-100'
                    : 'bg-white text-zinc-600 border-zinc-200'
                }`}
              >
                {c.highRisk && '⚠ '}
                {c.name}
              </span>
            ))}
          </div>
          <p className="text-zinc-300 text-xs mt-6">
            ⚠ Melanoma, BCC, and SCC are high-risk and trigger a dermatologist recommendation.
          </p>
        </div>
      </section>

      {/* About */}
      <section id="about" className="py-24 px-6 bg-zinc-50/50 border-y border-zinc-100">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-zinc-900">About SkinAI</h2>
            <p className="text-zinc-400 mt-4 text-lg">Who built it, why, and how it actually works.</p>
          </div>

          {/* Who + Why */}
          <div className="grid md:grid-cols-2 gap-10 mb-16">
            <div className="bg-white rounded-3xl border border-zinc-100 p-8">
              <div className="w-14 h-14 rounded-2xl bg-pink-50 border border-pink-100 flex items-center justify-center mb-5">
                <span className="text-xl font-bold text-pink-700">SL</span>
              </div>
              <h3 className="text-xl font-bold text-zinc-900 mb-1">Shawn (Huanxuan) Li</h3>
              <p className="text-pink-700 font-medium text-sm mb-4">Junior · TJHSST · Northern Virginia</p>
              <p className="text-zinc-500 leading-relaxed mb-4">
                I built SkinAI because early detection saves lives — and access to dermatologists shouldn&apos;t be limited
                by geography or cost. I have a personal connection to medicine; receiving life-saving treatment as a child showed me
                how critical early intervention is.
              </p>
              <p className="text-zinc-500 leading-relaxed">
                I also run <span className="font-medium text-zinc-700">CAPA</span> (a transplant immunology ML framework),{' '}
                <span className="font-medium text-zinc-700">Growing Up with Robotics</span> (a 501c3 nonprofit), and FTC robotics team 14607.
              </p>
            </div>

            <div className="bg-white rounded-3xl border border-zinc-100 p-8 flex flex-col justify-between">
              <div>
                <h3 className="text-xl font-bold text-zinc-900 mb-4">Why it exists</h3>
                <p className="text-zinc-500 leading-relaxed mb-4">
                  Skin cancer is the most common cancer in the US. Melanoma survival rate drops from 99% to 23% if caught late.
                  Most people don&apos;t see a dermatologist until something looks alarming — SkinAI closes that gap.
                </p>
                <p className="text-zinc-500 leading-relaxed">
                  With 20,000+ users and a pending patent, the dermatologist network layer is the core differentiator:
                  not just an AI flag, but a direct path to a real specialist.
                </p>
              </div>
              <div className="mt-6 flex gap-3">
                <a
                  href="https://github.com/sh4wn27/skinai/tree/main"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 border border-zinc-200 rounded-full text-sm font-medium text-zinc-600 hover:border-zinc-300 hover:text-zinc-900 transition-colors"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
                  </svg>
                  View on GitHub
                </a>
              </div>
            </div>
          </div>

          {/* Technical architecture */}
          <div className="bg-white rounded-3xl border border-zinc-100 p-8 mb-8">
            <h3 className="text-xl font-bold text-zinc-900 mb-6">How it&apos;s built</h3>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {[
                { label: 'Architecture', value: 'EfficientNet B4/B5/B7' },
                { label: 'Ensemble method', value: 'Soft-vote (mean prob.)' },
                { label: 'Training data', value: 'ISIC 2019 + 2020 + HAM10000' },
                { label: 'Output classes', value: '9 conditions' },
              ].map((m) => (
                <div key={m.label} className="bg-zinc-50 rounded-2xl p-4">
                  <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">{m.label}</div>
                  <div className="font-semibold text-zinc-800">{m.value}</div>
                </div>
              ))}
            </div>

            {/* Code snippet */}
            <div className="rounded-2xl overflow-hidden border border-zinc-200">
              <div className="bg-zinc-800 px-4 py-2.5 flex items-center justify-between">
                <span className="text-zinc-400 text-xs font-mono">backend/inference.py</span>
                <a
                  href="https://github.com/sh4wn27/skinai/tree/main"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-zinc-500 hover:text-zinc-300 text-xs transition-colors"
                >
                  view full source →
                </a>
              </div>
              <pre className="bg-zinc-900 text-zinc-300 p-5 text-sm font-mono overflow-x-auto leading-relaxed">
                <code>{`# Soft-voting ensemble across 3 EfficientNet models
def predict(ensemble, source, top_k=3):
    img = _open_image(source)
    probs = ensemble.predict_probs(img)   # mean of B4, B5, B7
    order = np.argsort(probs)[::-1][:top_k]

    return {
        "predictions": [
            {"class": CLASSES[i][1], "confidence": round(float(probs[i]), 4)}
            for i in order
        ],
        "high_risk": CLASSES[order[0]][0] in {"MEL", "BCC", "SCC"},
        "disclaimer": "For screening purposes only...",
    }

# ImageNet normalization (matches training preprocessing)
_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])
arr = (img_array / 255.0 - _MEAN) / _STD`}</code>
              </pre>
            </div>
          </div>

          {/* Stack badges */}
          <div className="flex flex-wrap gap-3 justify-center">
            {['Python 3.13', 'TensorFlow 2.20', 'FastAPI', 'Next.js 14', 'Tailwind CSS', 'Vercel', 'Modal (GPU)'].map((t) => (
              <span key={t} className="px-4 py-2 bg-white border border-zinc-200 rounded-full text-sm font-medium text-zinc-600">
                {t}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* Dermatologist waitlist */}
      <section className="py-24 px-6 bg-gradient-to-br from-pink-50 to-rose-50 border-y border-pink-100">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-zinc-900 mb-4">Are you a dermatologist?</h2>
          <p className="text-zinc-500 text-lg mb-10 leading-relaxed">
            Join our verified network and connect with patients who need expert evaluation after a
            high-risk screening result.
          </p>
          <a
            href="mailto:shawnli1028@gmail.com?subject=SkinAI%20Dermatologist%20Network%20Application"
            className="inline-block px-10 py-4 bg-pink-700 text-white rounded-full font-semibold hover:bg-pink-800 transition-colors text-lg shadow-lg shadow-pink-200"
          >
            Apply to Join →
          </a>
          <p className="text-zinc-400 text-sm mt-6">10+ dermatologists already onboarded</p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-zinc-100">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-pink-700 flex items-center justify-center">
              <span className="text-white font-bold text-xs">S</span>
            </div>
            <span className="font-bold text-zinc-900">SkinAI</span>
          </div>

          <p className="text-zinc-400 text-sm text-center max-w-md">
            For screening purposes only. Not a substitute for professional medical advice. Always
            consult a qualified dermatologist.
          </p>

          <div className="flex gap-6 text-sm text-zinc-400">
            <Link href="/scan" className="hover:text-zinc-700 transition-colors">
              Scan
            </Link>
            <Link href="/dermatologists" className="hover:text-zinc-700 transition-colors">
              Dermatologists
            </Link>
          </div>
        </div>
        <p className="text-zinc-300 text-xs text-center mt-6">
          Built by Shawn Li · TJHSST · Pending Patent
        </p>
      </footer>
    </main>
  )
}
