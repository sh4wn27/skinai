import type { Metadata } from 'next'
import Link from 'next/link'
import Nav from '@/components/Nav'
import ScanWidget from '@/components/ScanWidget'

export const metadata: Metadata = {
  title: 'Scan — SkinAI',
  description: 'Upload a photo for AI-powered skin lesion screening across 9 conditions.',
}

export default function ScanPage() {
  return (
    <main className="min-h-screen bg-white">
      <Nav />

      <div className="pt-28 pb-24 px-6">
        <div className="max-w-2xl mx-auto">
          <div className="mb-10">
            <Link
              href="/"
              className="text-sm text-zinc-400 hover:text-zinc-600 transition-colors flex items-center gap-1 mb-6"
            >
              ← Back
            </Link>
            <h1 className="text-4xl font-bold text-zinc-900 mb-2">Scan your skin</h1>
            <p className="text-zinc-400 text-lg">
              Upload a clear, well-lit photo of the lesion. Our AI will screen it for 9 conditions
              in seconds.
            </p>
          </div>

          <ScanWidget showDisclaimer={true} />
        </div>
      </div>

      {/* Sticky disclaimer bar */}
      <div className="fixed bottom-0 inset-x-0 bg-white/90 backdrop-blur-sm border-t border-zinc-100 py-2.5 px-6">
        <p className="text-xs text-zinc-400 text-center">
          For screening purposes only. Not a substitute for professional medical advice. Always
          consult a qualified dermatologist.
        </p>
      </div>
    </main>
  )
}
