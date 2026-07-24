'use client'

import Script from 'next/script'
import { useEffect, useRef } from 'react'

export interface DermResult {
  id: string
  name: string
  address: string | null
  lat: number | null
  lng: number | null
  distance_km: number | null
  rating: number | null
  rating_count: number | null
  phone: string | null
  maps_url: string | null
}

interface Props {
  userLocation: { lat: number; lng: number }
  results: DermResult[]
  selectedId: string | null
  onSelect: (id: string) => void
}

declare global {
  interface Window {
    google?: typeof google
  }
}

export default function DermatologistMap({ userLocation, results, selectedId, onSelect }: Props) {
  const mapDivRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<google.maps.Map | null>(null)
  const markersRef = useRef<Map<string, google.maps.Marker>>(new Map())
  const scriptLoadedRef = useRef(false)

  const initMap = () => {
    if (!mapDivRef.current || !window.google || mapRef.current) return
    mapRef.current = new window.google.maps.Map(mapDivRef.current, {
      center: userLocation,
      zoom: 12,
      disableDefaultUI: false,
      streetViewControl: false,
      mapTypeControl: false,
    })
    new window.google.maps.Marker({
      position: userLocation,
      map: mapRef.current,
      title: 'Your location',
      icon: {
        path: window.google.maps.SymbolPath.CIRCLE,
        scale: 8,
        fillColor: '#be185d',
        fillOpacity: 1,
        strokeColor: '#ffffff',
        strokeWeight: 2,
      },
    })
    scriptLoadedRef.current = true
    syncMarkers()
  }

  const syncMarkers = () => {
    const map = mapRef.current
    if (!map || !window.google) return

    markersRef.current.forEach((marker) => marker.setMap(null))
    markersRef.current.clear()

    for (const r of results) {
      if (r.lat == null || r.lng == null) continue
      const marker = new window.google.maps.Marker({
        position: { lat: r.lat, lng: r.lng },
        map,
        title: r.name,
      })
      marker.addListener('click', () => onSelect(r.id))
      markersRef.current.set(r.id, marker)
    }
  }

  useEffect(() => {
    if (scriptLoadedRef.current) syncMarkers()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results])

  useEffect(() => {
    const map = mapRef.current
    const marker = selectedId ? markersRef.current.get(selectedId) : null
    if (map && marker) {
      map.panTo(marker.getPosition()!)
      map.setZoom(Math.max(map.getZoom() ?? 12, 14))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId])

  return (
    <>
      <Script
        src={`https://maps.googleapis.com/maps/api/js?key=${process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY}`}
        strategy="afterInteractive"
        onLoad={initMap}
      />
      <div
        ref={mapDivRef}
        className="w-full h-full min-h-[320px] rounded-2xl border border-zinc-100 bg-zinc-50"
      />
    </>
  )
}
