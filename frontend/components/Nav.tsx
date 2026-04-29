import Link from 'next/link'

export default function Nav() {
  return (
    <nav className="fixed top-0 inset-x-0 z-40 bg-white/80 backdrop-blur-md border-b border-zinc-100">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-pink-700 flex items-center justify-center">
            <span className="text-white font-semibold text-xs">S</span>
          </div>
          <span className="font-semibold text-zinc-900 text-sm tracking-tight">SkinAI</span>
        </Link>

        <div className="flex items-center gap-4 sm:gap-6">
          <Link href="/scan" className="text-xs text-zinc-500 hover:text-zinc-900 transition-colors font-medium hidden sm:block">
            Scan
          </Link>
          <Link href="/dermatologists" className="text-xs text-zinc-500 hover:text-zinc-900 transition-colors font-medium hidden sm:block">
            Dermatologists
          </Link>
          <Link
            href="/scan"
            className="px-3.5 py-1.5 bg-pink-700 text-white rounded-full font-medium hover:bg-pink-800 transition-colors text-xs"
          >
            Get Screened
          </Link>
        </div>
      </div>
    </nav>
  )
}
