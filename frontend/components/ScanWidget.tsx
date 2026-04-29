'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import Link from 'next/link'

type Prediction = {
  class: string
  code: string
  confidence: number
}

type PredictResult = {
  predictions: Prediction[]
  high_risk: boolean
  top_class: string
  disclaimer: string
}

interface ScanWidgetProps {
  showDisclaimer?: boolean
}

export default function ScanWidget({ showDisclaimer = false }: ScanWidgetProps) {
  const [disclaimerAccepted, setDisclaimerAccepted] = useState(true)
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (showDisclaimer) {
      const accepted = localStorage.getItem('skinai_disclaimer_accepted')
      if (!accepted) setDisclaimerAccepted(false)
    }
  }, [showDisclaimer])

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG, or WEBP).')
      return
    }
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const f = e.dataTransfer.files[0]
      if (f) handleFile(f)
    },
    [handleFile],
  )

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  const toJpeg = (f: File): Promise<Blob> =>
    new Promise((resolve, reject) => {
      const img = new Image()
      const url = URL.createObjectURL(f)
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        canvas.getContext('2d')!.drawImage(img, 0, 0)
        URL.revokeObjectURL(url)
        canvas.toBlob(
          (blob) => (blob ? resolve(blob) : reject(new Error('canvas conversion failed'))),
          'image/jpeg',
          0.92,
        )
      }
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('could not load image')) }
      img.src = url
    })

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const jpeg = await toJpeg(file)
      const formData = new FormData()
      formData.append('file', jpeg, 'image.jpg')
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const res = await fetch(`${apiUrl}/predict`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error(`Server error (${res.status})`)
      const data: PredictResult = await res.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setResult(null)
    setPreview(null)
    setFile(null)
    setError(null)
  }

  if (!disclaimerAccepted) {
    return (
      <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4">
        <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl">
          <div className="w-12 h-12 rounded-2xl bg-amber-50 flex items-center justify-center mb-6">
            <svg className="w-6 h-6 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold mb-3">Medical Disclaimer</h2>
          <p className="text-zinc-500 mb-4 leading-relaxed">
            SkinAI is a screening tool powered by artificial intelligence. It is{' '}
            <strong className="text-zinc-900">not a medical device</strong> and does not provide
            medical diagnoses.
          </p>
          <p className="text-zinc-500 mb-8 leading-relaxed">
            Results are for informational purposes only and do not replace professional medical
            evaluation. Always consult a qualified dermatologist for diagnosis and treatment.
          </p>
          <button
            onClick={() => {
              localStorage.setItem('skinai_disclaimer_accepted', '1')
              setDisclaimerAccepted(true)
            }}
            className="w-full py-4 bg-pink-700 text-white rounded-2xl font-semibold hover:bg-pink-800 transition-colors text-lg"
          >
            I Understand
          </button>
        </div>
      </div>
    )
  }

  if (result) {
    return (
      <div className="w-full max-w-2xl mx-auto space-y-5">
        <div
          className={`rounded-3xl p-6 border-2 ${
            result.high_risk
              ? 'bg-red-50 border-red-200'
              : 'bg-emerald-50 border-emerald-200'
          }`}
        >
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold mb-3 ${
              result.high_risk ? 'bg-red-500 text-white' : 'bg-emerald-500 text-white'
            }`}
          >
            {result.high_risk ? '⚠ High Risk' : '✓ Lower Risk'}
          </span>
          <h3 className="text-2xl font-bold text-zinc-900">{result.top_class}</h3>
          <p className="text-zinc-500 text-sm mt-2 leading-relaxed">{result.disclaimer}</p>
        </div>

        <div className="bg-zinc-50 rounded-3xl p-6 space-y-5">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
            Top Predictions
          </h4>
          {result.predictions.map((p, i) => (
            <div key={p.code}>
              <div className="flex justify-between items-baseline mb-1.5">
                <span className="font-semibold text-zinc-800">{p.class}</span>
                <span className="text-sm text-zinc-400 font-mono tabular-nums">
                  {(p.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-zinc-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${
                    i === 0 ? 'bg-pink-700' : 'bg-pink-300'
                  }`}
                  style={{ width: `${p.confidence * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {result.high_risk && (
          <div className="bg-pink-50 border border-pink-100 rounded-3xl p-6 text-center">
            <p className="font-semibold text-zinc-800 mb-1">Consider seeing a specialist</p>
            <p className="text-zinc-500 text-sm mb-4">
              High-risk findings benefit from in-person evaluation.
            </p>
            <Link
              href="/dermatologists"
              className="inline-block px-6 py-3 bg-pink-700 text-white rounded-full font-semibold hover:bg-pink-800 transition-colors text-sm"
            >
              View Our Dermatologists
            </Link>
          </div>
        )}

        <button
          onClick={reset}
          className="w-full py-3 border-2 border-zinc-200 text-zinc-600 rounded-2xl font-semibold hover:border-zinc-300 transition-colors"
        >
          Scan another image
        </button>
      </div>
    )
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      <input
        type="file"
        accept="image/*"
        ref={fileRef}
        className="hidden"
        onChange={handleChange}
      />

      {!preview ? (
        <div
          className={`border-2 border-dashed rounded-3xl p-16 text-center cursor-pointer transition-all select-none ${
            isDragging
              ? 'border-pink-500 bg-pink-50'
              : 'border-zinc-200 hover:border-pink-300 hover:bg-pink-50/40'
          }`}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          role="button"
          aria-label="Upload image"
        >
          <div className="w-16 h-16 rounded-2xl bg-pink-50 border border-pink-100 flex items-center justify-center mx-auto mb-5">
            <svg
              className="w-8 h-8 text-pink-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
          </div>
          <p className="text-lg font-semibold text-zinc-700">Drop your photo here</p>
          <p className="text-zinc-400 mt-1 text-sm">or click to browse</p>
          <p className="text-zinc-300 text-xs mt-4 font-light">JPG, PNG, WEBP · up to 10 MB</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-3xl overflow-hidden bg-zinc-50 flex items-center justify-center min-h-64 max-h-80">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="Preview" className="max-h-80 object-contain" />
            <button
              onClick={reset}
              className="absolute top-3 right-3 w-8 h-8 bg-white/90 rounded-full flex items-center justify-center hover:bg-white transition-colors shadow-sm"
              aria-label="Remove image"
            >
              <svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full py-4 bg-pink-700 text-white rounded-2xl font-semibold hover:bg-pink-800 active:bg-pink-900 transition-colors disabled:opacity-60 disabled:cursor-not-allowed text-lg"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing… (first use may take 30–60s)
              </span>
            ) : (
              'Analyze Image'
            )}
          </button>
        </div>
      )}

      {error && (
        <p className="mt-4 text-red-500 text-sm text-center">{error}</p>
      )}
    </div>
  )
}
