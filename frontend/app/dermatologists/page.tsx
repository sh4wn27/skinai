import type { Metadata } from 'next'
import Link from 'next/link'
import Nav from '@/components/Nav'
import DermCard from '@/components/DermCard'
import derms from '@/data/dermatologists.json'

export const metadata: Metadata = {
  title: 'Dermatologists — SkinAI',
  description: 'Our verified network of dermatologists available for consultation.',
}

export default function DermatologistsPage() {
  return (
    <main className="min-h-screen bg-white">
      <Nav />

      <div className="pt-28 pb-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="mb-12">
            <Link
              href="/"
              className="text-sm text-zinc-400 hover:text-zinc-600 transition-colors flex items-center gap-1 mb-6"
            >
              ← Back
            </Link>
            <h1 className="text-4xl font-bold text-zinc-900 mb-2">Our Dermatologists</h1>
            <p className="text-zinc-400 text-lg">
              Verified specialists ready for consultation. Click any card to request an
              appointment.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {derms.map((d) => (
              <DermCard key={d.id} {...d} />
            ))}
          </div>

          {/* Join CTA */}
          <div className="mt-16 rounded-3xl bg-gradient-to-br from-pink-50 to-pink-50 border border-pink-100 p-10 text-center">
            <h2 className="text-2xl font-bold text-zinc-900 mb-3">Not on the list?</h2>
            <p className="text-zinc-500 mb-6">
              We're actively onboarding verified dermatologists across the US.
            </p>
            <a
              href="mailto:shawnli1028@gmail.com?subject=SkinAI%20Dermatologist%20Network%20Application"
              className="inline-block px-8 py-3 bg-pink-700 text-white rounded-full font-semibold hover:bg-pink-800 transition-colors"
            >
              Apply to Join →
            </a>
          </div>
        </div>
      </div>

      <footer className="border-t border-zinc-100 py-8 px-6">
        <p className="text-xs text-zinc-400 text-center">
          For screening purposes only. Not a substitute for professional medical advice.
        </p>
      </footer>
    </main>
  )
}
