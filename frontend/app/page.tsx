import Link from 'next/link'
import Nav from '@/components/Nav'
import ScanWidget from '@/components/ScanWidget'

const stats = [
  { value: '25,000+', label: 'Training Images' },
  { value: 'B4 · B5 · B7', label: 'EfficientNet Ensemble' },
  { value: '9', label: 'Conditions Detected' },
]

const steps = [
  {
    title: 'Upload',
    desc: 'Take or upload a clear photo of the skin lesion.',
  },
  {
    title: 'Analyze',
    desc: 'EfficientNet ensemble screens for 9 conditions in seconds.',
  },
  {
    title: 'Connect',
    desc: 'High-risk results are clearly flagged so you can seek professional evaluation.',
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

const modelFacts = [
  { label: 'Architecture', value: 'EfficientNet B4 · B5 · B7' },
  { label: 'Ensemble', value: 'Soft-vote (mean probs)' },
  { label: 'Training data', value: 'ISIC 2019 + 2020 + HAM10000' },
  { label: 'Training images', value: '25,000+' },
  { label: 'Output classes', value: '9 conditions' },
  { label: 'High-risk flags', value: 'MEL · BCC · SCC' },
]

export default function HomePage() {
  return (
    <main className="min-h-screen bg-white">
      <Nav />

      {/* Hero */}
      <section className="pt-28 pb-16 px-4 sm:px-6 text-center">
        <div className="max-w-3xl mx-auto">
<h1 className="text-3xl sm:text-5xl md:text-6xl font-semibold tracking-tight text-zinc-900 mb-4 leading-[1.1]">
            AI Skin Cancer
            <br />
            <span className="text-pink-700">Screening.</span>
          </h1>
          <p className="text-sm sm:text-base text-zinc-400 max-w-xl mx-auto mb-8 leading-relaxed font-light">
            Upload a photo of your skin lesion. Get an instant risk assessment across 9
            dermatological conditions powered by a research-grade EfficientNet ensemble.
          </p>
          <div className="flex gap-3 justify-center flex-wrap">
            <Link
              href="/scan"
              className="px-6 py-3 bg-pink-700 text-white rounded-full font-medium hover:bg-pink-800 transition-colors text-sm shadow-md shadow-pink-100"
            >
              Get Screened Free
            </Link>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-8 border-y border-zinc-100 bg-zinc-50/60">
        <div className="max-w-2xl mx-auto grid grid-cols-3 gap-4 text-center px-4">
          {stats.map((s) => (
            <div key={s.label}>
              <div className="text-2xl sm:text-3xl font-semibold text-pink-700">{s.value}</div>
              <div className="text-zinc-400 mt-1 text-xs font-light">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Try it live */}
      <section className="py-16 px-4 sm:px-6">
        <div className="max-w-xl mx-auto text-center mb-8">
          <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">Live Demo</p>
          <h2 className="text-2xl sm:text-3xl font-semibold text-zinc-900 mb-3">Try it now</h2>
          <p className="text-zinc-400 text-sm font-light">
            Upload a photo and see the AI analyze it in real time.
          </p>
        </div>
        <ScanWidget showDisclaimer={false} />
      </section>

      {/* How it works */}
      <section className="py-16 px-4 sm:px-6 bg-zinc-50/60 border-y border-zinc-100">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">Process</p>
            <h2 className="text-2xl sm:text-3xl font-semibold text-zinc-900">How it works</h2>
          </div>
          <div className="grid sm:grid-cols-3 gap-6 sm:gap-10">
            {steps.map((step, i) => (
              <div key={i} className="text-center">
                <div className="w-10 h-10 rounded-xl bg-pink-700 flex items-center justify-center mx-auto mb-4">
                  <span className="text-sm font-semibold text-white">{i + 1}</span>
                </div>
                <h3 className="text-sm font-semibold text-zinc-900 mb-2">{step.title}</h3>
                <p className="text-zinc-400 text-xs leading-relaxed font-light">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Conditions */}
      <section className="py-16 px-4 sm:px-6">
        <div className="max-w-3xl mx-auto text-center">
          <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">Coverage</p>
          <h2 className="text-2xl sm:text-3xl font-semibold text-zinc-900 mb-2">9 conditions detected</h2>
          <p className="text-zinc-400 text-xs font-light mb-8">
            Trained on ISIC 2019 + ISIC 2020 + HAM10000
          </p>
          <div className="flex flex-wrap gap-2 justify-center">
            {conditions.map((c) => (
              <span
                key={c.code}
                className={`px-3 py-1.5 rounded-full text-xs font-medium border ${
                  c.highRisk
                    ? 'bg-pink-50 text-pink-700 border-pink-200'
                    : 'bg-white text-zinc-500 border-zinc-200'
                }`}
              >
                {c.name}
              </span>
            ))}
          </div>
          <p className="text-zinc-300 text-xs mt-5 font-light">
            Melanoma, BCC, and SCC are flagged high-risk and trigger a dermatologist recommendation.
          </p>
        </div>
      </section>

      {/* Model */}
      <section className="py-16 px-4 sm:px-6 bg-zinc-50/60 border-y border-zinc-100">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">Under the Hood</p>
            <h2 className="text-2xl sm:text-3xl font-semibold text-zinc-900 mb-2">Research-grade model</h2>
            <p className="text-zinc-400 text-sm font-light max-w-xl mx-auto">
              A soft-voting ensemble of three EfficientNet models, each trained independently and averaged at inference time.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-8">
            {modelFacts.map((f) => (
              <div key={f.label} className="bg-white border border-zinc-100 rounded-2xl p-4">
                <div className="text-xs text-pink-600 font-medium uppercase tracking-wider mb-1">{f.label}</div>
                <div className="text-sm font-medium text-zinc-800">{f.value}</div>
              </div>
            ))}
          </div>

          <div className="rounded-2xl overflow-hidden border border-zinc-200">
            <div className="bg-zinc-800 px-4 py-2 flex items-center justify-between">
              <span className="text-zinc-400 text-xs font-light">inference.py</span>
              <a
                href="https://github.com/sh4wn27/skinai/tree/main"
                target="_blank"
                rel="noopener noreferrer"
                className="text-pink-400 hover:text-pink-300 text-xs transition-colors"
              >
                view source →
              </a>
            </div>
            <pre className="bg-zinc-900 text-zinc-300 p-4 text-xs font-mono overflow-x-auto leading-relaxed">
              <code>{`# Soft-voting ensemble
def predict_probs(self, img):
    per_model = []
    for model in self.models:          # B4, B5, B7
        h, w = self.input_size(model)
        arr  = _preprocess(img, (h, w))
        per_model.append(model.predict(arr)[0])
    return np.mean(per_model, axis=0)  # soft vote

# ImageNet normalization
arr = (img_array / 255.0 - [0.485,0.456,0.406]) \
                         / [0.229,0.224,0.225]`}</code>
            </pre>
          </div>

          <div className="flex flex-wrap gap-2 justify-center mt-6">
            {['Python 3.11', 'TensorFlow 2.20', 'tf-keras', 'FastAPI', 'Next.js 14', 'Modal'].map((t) => (
              <span key={t} className="px-3 py-1 bg-white border border-zinc-200 rounded-full text-xs font-light text-zinc-500">
                {t}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* Dermatologist waitlist */}
      <section className="py-16 px-4 sm:px-6">
        <div className="max-w-xl mx-auto text-center">
          <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">Network</p>
          <h2 className="text-2xl sm:text-3xl font-semibold text-zinc-900 mb-3">Are you a dermatologist?</h2>
          <p className="text-zinc-400 text-sm font-light mb-8 leading-relaxed">
            Join our verified network and connect with patients who need expert evaluation after a
            high-risk screening result.
          </p>
          <a
            href="mailto:shawnli1028@gmail.com?subject=SkinAI%20Dermatologist%20Network%20Application"
            className="inline-block px-7 py-3 bg-pink-700 text-white rounded-full font-medium hover:bg-pink-800 transition-colors text-sm shadow-md shadow-pink-100"
          >
            Apply to Join →
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 sm:px-6 border-t border-zinc-100">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-pink-700 flex items-center justify-center">
              <span className="text-white font-semibold text-xs">S</span>
            </div>
            <span className="font-medium text-zinc-900 text-sm">SkinAI</span>
          </div>

          <p className="text-zinc-400 text-xs text-center font-light max-w-sm">
            For screening purposes only. Not a substitute for professional medical advice.
          </p>

          <div className="flex items-center gap-4 text-xs text-zinc-400 font-light">
            <Link href="/scan" className="hover:text-pink-700 transition-colors">Scan</Link>
<a
              href="https://github.com/sh4wn27/skinai"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-pink-700 transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
        <p className="text-zinc-300 text-xs text-center mt-4 font-light">
          Built with ❤️ by Shawn. © 2026 SkinAI. All rights reserved.
        </p>
      </footer>
    </main>
  )
}
