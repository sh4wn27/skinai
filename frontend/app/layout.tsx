import type { Metadata } from 'next'
import { Sora } from 'next/font/google'
import './globals.css'

const sora = Sora({
  subsets: ['latin'],
  variable: '--font-sora',
  weight: ['300', '400', '500', '600'],
})

export const metadata: Metadata = {
  title: 'SkinAI — AI Skin Cancer Screening',
  description:
    'AI-powered skin lesion screening across 9 dermatological conditions. For screening purposes only. Not a substitute for professional medical advice.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={sora.variable}>
      <body className="bg-white text-zinc-900 antialiased">{children}</body>
    </html>
  )
}
