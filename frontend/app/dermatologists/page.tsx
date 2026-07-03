import type { Metadata } from 'next'
import Link from 'next/link'
import Nav from '@/components/Nav'

export const metadata: Metadata = {
  title: 'Dermatologists — SkinAI',
  description:
    'Find a board-certified dermatologist for follow-up after a high-risk skin lesion screening.',
}

const resources = [
  {
    name: 'American Academy of Dermatology',
    desc: 'Find a board-certified dermatologist near you.',
    url: 'https://www.aad.org/public/fad',
    label: 'aad.org',
  },
  {
    name: 'Skin Cancer Foundation',
    desc: 'Education, prevention, and a specialist finder.',
    url: 'https://www.skincancer.org/find-a-dermatologist/',
    label: 'skincancer.org',
  },
  {
    name: 'American Cancer Society',
    desc: 'Skin cancer information and clinical trial navigator.',
    url: 'https://www.cancer.org/cancer/melanoma-skin-cancer.html',
    label: 'cancer.org',
  },
]

export default function DermatologistsPage() {
  return (
    <main className="min-h-screen bg-white">
      <Nav />

      {/* Hero */}
      <section className="pt-28 pb-12 px-4 sm:px-6 text-center">
        <div className="max-w-2xl mx-auto">
          <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">
            Network
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold text-zinc-900 mb-3 tracking-tight">
            Verified Dermatologists
          </h1>
          <p className="text-zinc-400 text-sm font-light leading-relaxed max-w-lg mx-auto">
            Connect with board-certified specialists for in-person evaluation after a high-risk
            screening result.
          </p>
        </div>
      </section>

      {/* Empty state */}
      <section className="py-8 px-4 sm:px-6">
        <div className="max-w-xl mx-auto">
          <div className="rounded-3xl border-2 border-dashed border-zinc-200 p-12 text-center">
            <div className="w-14 h-14 rounded-2xl bg-pink-50 flex items-center justify-center mx-auto mb-5">
              <svg
                className="w-7 h-7 text-pink-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            </div>
            <h2 className="text-lg font-semibold text-zinc-800 mb-2">Network launching soon</h2>
            <p className="text-zinc-400 text-sm font-light leading-relaxed mb-6 max-w-xs mx-auto">
              We&apos;re onboarding board-certified dermatologists who specialize in skin cancer
              detection. Check back soon.
            </p>
            <a
              href="https://www.aad.org/public/fad"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block px-6 py-3 bg-pink-700 text-white rounded-full font-medium hover:bg-pink-800 transition-colors text-sm shadow-md shadow-pink-100"
            >
              Find a Dermatologist via AAD →
            </a>
            <p className="text-zinc-300 text-xs mt-4 font-light">
              American Academy of Dermatology · free locator tool
            </p>
          </div>
        </div>
      </section>

      {/* Resources */}
      <section className="py-14 px-4 sm:px-6 bg-zinc-50/60 border-y border-zinc-100">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-8">
            <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">
              Resources
            </p>
            <h2 className="text-xl sm:text-2xl font-semibold text-zinc-900">
              Trusted skin cancer resources
            </h2>
          </div>
          <div className="grid sm:grid-cols-3 gap-4">
            {resources.map((r) => (
              <a
                key={r.name}
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-white rounded-2xl border border-zinc-100 p-5 hover:border-pink-200 hover:shadow-sm transition-all group"
              >
                <div className="text-xs text-pink-600 font-medium mb-1 group-hover:text-pink-700">
                  {r.label}
                </div>
                <div className="font-semibold text-zinc-800 text-sm mb-1">{r.name}</div>
                <div className="text-zinc-400 text-xs font-light leading-relaxed">{r.desc}</div>
              </a>
            ))}
          </div>
        </div>
      </section>

      {/* Join CTA */}
      <section className="py-16 px-4 sm:px-6">
        <div className="max-w-xl mx-auto text-center">
          <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">Join</p>
          <h2 className="text-2xl sm:text-3xl font-semibold text-zinc-900 mb-3">
            Are you a dermatologist?
          </h2>
          <p className="text-zinc-400 text-sm font-light mb-8 leading-relaxed">
            Join our verified network and connect with patients who need expert evaluation after a
            high-risk screening result. Board certification required.
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
            <Link href="/scan" className="hover:text-pink-700 transition-colors">
              Scan
            </Link>
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
