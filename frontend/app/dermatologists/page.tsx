'use client'

import { useState } from 'react'
import Link from 'next/link'
import Nav from '@/components/Nav'
import DermatologistMap, { type DermResult } from '@/components/DermatologistMap'

const API_URL = process.env.NEXT_PUBLIC_API_URL

type LocationState =
  | { status: 'idle' }
  | { status: 'locating' }
  | { status: 'found'; lat: number; lng: number }
  | { status: 'error'; message: string }

export default function DermatologistsPage() {
  const [location, setLocation] = useState<LocationState>({ status: 'idle' })
  const [manualAddress, setManualAddress] = useState('')
  const [results, setResults] = useState<DermResult[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loadingResults, setLoadingResults] = useState(false)
  const [resultsError, setResultsError] = useState<string | null>(null)

  async function fetchNearby(lat: number, lng: number) {
    setLoadingResults(true)
    setResultsError(null)
    try {
      const res = await fetch(
        `${API_URL}/api/dermatologists/nearby?lat=${lat}&lng=${lng}`
      )
      if (!res.ok) throw new Error(`request failed (${res.status})`)
      const data = await res.json()
      setResults(data.results ?? [])
    } catch {
      setResultsError("Couldn't load nearby dermatologists. Please try again.")
    } finally {
      setLoadingResults(false)
    }
  }

  function useBrowserLocation() {
    if (!('geolocation' in navigator)) {
      setLocation({ status: 'error', message: 'Location is not available in this browser.' })
      return
    }
    setLocation({ status: 'locating' })
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords
        setLocation({ status: 'found', lat: latitude, lng: longitude })
        fetchNearby(latitude, longitude)
      },
      () => {
        setLocation({
          status: 'error',
          message: "Couldn't access your location. Enter an address or zip code instead.",
        })
      },
      { timeout: 10000 }
    )
  }

  async function submitManualAddress(e: React.FormEvent) {
    e.preventDefault()
    if (!manualAddress.trim()) return
    setLocation({ status: 'locating' })
    try {
      const res = await fetch(`${API_URL}/api/geocode?address=${encodeURIComponent(manualAddress)}`)
      if (!res.ok) throw new Error()
      const data = await res.json()
      setLocation({ status: 'found', lat: data.lat, lng: data.lng })
      fetchNearby(data.lat, data.lng)
    } catch {
      setLocation({
        status: 'error',
        message: "Couldn't find that address. Try a more specific address or zip code.",
      })
    }
  }

  return (
    <main className="min-h-screen bg-white">
      <Nav />

      {/* Hero */}
      <section className="pt-28 pb-10 px-4 sm:px-6 text-center">
        <div className="max-w-2xl mx-auto">
          <p className="text-xs font-medium text-pink-700 uppercase tracking-widest mb-2">
            Find Care
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold text-zinc-900 mb-3 tracking-tight">
            Find a Dermatologist Near You
          </h1>
          <p className="text-zinc-400 text-sm font-light leading-relaxed max-w-lg mx-auto">
            Search live results from Google to find dermatologist offices near you for
            in-person evaluation after a screening result. SkinAI does not endorse or verify
            any listed provider — always confirm credentials directly.
          </p>
        </div>
      </section>

      {/* Location entry */}
      <section className="px-4 sm:px-6 pb-8">
        <div className="max-w-xl mx-auto flex flex-col items-center gap-4">
          <button
            onClick={useBrowserLocation}
            disabled={location.status === 'locating'}
            className="px-6 py-3 bg-pink-700 text-white rounded-full font-medium hover:bg-pink-800 transition-colors text-sm shadow-md shadow-pink-100 disabled:opacity-60"
          >
            {location.status === 'locating' ? 'Locating…' : 'Find Dermatologists Near Me'}
          </button>

          {location.status === 'error' && (
            <p className="text-xs text-zinc-500">{location.message}</p>
          )}

          <form onSubmit={submitManualAddress} className="flex w-full gap-2">
            <input
              type="text"
              value={manualAddress}
              onChange={(e) => setManualAddress(e.target.value)}
              placeholder="Or enter an address or zip code"
              className="flex-1 px-4 py-2.5 rounded-full border border-zinc-200 text-sm focus:outline-none focus:border-pink-400"
            />
            <button
              type="submit"
              disabled={location.status === 'locating'}
              className="px-5 py-2.5 rounded-full border border-zinc-200 text-sm font-medium text-zinc-700 hover:border-pink-300 transition-colors disabled:opacity-60"
            >
              Search
            </button>
          </form>
        </div>
      </section>

      {/* Results */}
      {location.status === 'found' && (
        <section className="px-4 sm:px-6 pb-16">
          <div className="max-w-5xl mx-auto grid md:grid-cols-2 gap-6">
            <div className="h-[420px] md:h-[520px]">
              <DermatologistMap
                userLocation={{ lat: location.lat, lng: location.lng }}
                results={results}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </div>

            <div className="flex flex-col gap-3 max-h-[520px] overflow-y-auto pr-1">
              {loadingResults && (
                <p className="text-sm text-zinc-400 font-light">Searching nearby…</p>
              )}
              {resultsError && <p className="text-sm text-zinc-500">{resultsError}</p>}
              {!loadingResults && !resultsError && results.length === 0 && (
                <p className="text-sm text-zinc-400 font-light">
                  No dermatologist offices found nearby. Try a broader search.
                </p>
              )}
              {results.map((r) => (
                <button
                  key={r.id}
                  onClick={() => setSelectedId(r.id)}
                  className={`text-left rounded-2xl border p-4 transition-all ${
                    selectedId === r.id
                      ? 'border-pink-300 bg-pink-50/60'
                      : 'border-zinc-100 hover:border-pink-200'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-semibold text-zinc-800 text-sm">{r.name}</div>
                    {r.distance_km != null && (
                      <div className="text-xs text-pink-700 font-medium whitespace-nowrap">
                        {r.distance_km} km
                      </div>
                    )}
                  </div>
                  {r.address && (
                    <div className="text-zinc-400 text-xs font-light mt-1">{r.address}</div>
                  )}
                  <div className="flex items-center gap-3 mt-2 text-xs text-zinc-400 font-light">
                    {r.rating != null && (
                      <span>
                        ★ {r.rating} ({r.rating_count ?? 0})
                      </span>
                    )}
                    {r.phone && <span>{r.phone}</span>}
                  </div>
                  {r.lat != null && r.lng != null && (
                    <a
                      href={
                        r.maps_url ??
                        `https://www.google.com/maps/dir/?api=1&destination=${r.lat},${r.lng}`
                      }
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="inline-block mt-2 text-xs font-medium text-pink-700 hover:text-pink-800"
                    >
                      Get Directions →
                    </a>
                  )}
                </button>
              ))}
            </div>
          </div>
        </section>
      )}

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
            {[
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
            ].map((r) => (
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
