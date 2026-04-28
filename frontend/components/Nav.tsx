import Link from 'next/link'

export default function Nav() {
  return (
    <nav className="fixed top-0 inset-x-0 z-40 bg-white/80 backdrop-blur-md border-b border-zinc-100">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-7 h-7 rounded-lg bg-pink-700 flex items-center justify-center">
            <span className="text-white font-bold text-xs">S</span>
          </div>
          <span className="font-bold text-zinc-900 font-heading">SkinAI</span>
        </Link>

        <div className="flex items-center gap-6 md:gap-8">
          <Link
            href="/scan"
            className="text-sm text-zinc-500 hover:text-zinc-900 transition-colors font-medium"
          >
            Scan
          </Link>
          <Link
            href="/dermatologists"
            className="text-sm text-zinc-500 hover:text-zinc-900 transition-colors font-medium"
          >
            Dermatologists
          </Link>
          <Link
            href="/#about"
            className="text-sm text-zinc-500 hover:text-zinc-900 transition-colors font-medium"
          >
            About
          </Link>
          <Link
            href="/scan"
            className="px-4 py-2 bg-pink-700 text-white rounded-full font-semibold hover:bg-pink-800 transition-colors text-sm shadow-sm shadow-pink-200"
          >
            Get Screened
          </Link>
        </div>
      </div>
    </nav>
  )
}
