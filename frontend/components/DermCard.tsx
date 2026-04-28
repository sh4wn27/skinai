interface DermCardProps {
  id: string
  name: string
  specialty: string
  subspecialty?: string
  location: string
  hospital: string
  email: string
  verified: boolean
}

export default function DermCard({
  name,
  specialty,
  subspecialty,
  location,
  hospital,
  email,
  verified,
}: DermCardProps) {
  const initials = name
    .replace('Dr. ', '')
    .split(' ')
    .map((n) => n[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

  return (
    <div className="bg-white rounded-3xl border border-zinc-100 p-6 hover:border-pink-100 hover:shadow-lg hover:shadow-pink-50 transition-all duration-200 flex flex-col">
      <div className="w-14 h-14 rounded-2xl bg-pink-50 border border-pink-100 flex items-center justify-center mb-4 shrink-0">
        <span className="text-xl font-bold text-pink-700">{initials}</span>
      </div>

      <div className="flex-1">
        <h3 className="font-bold text-lg text-zinc-900 leading-tight">{name}</h3>
        <p className="text-pink-700 font-medium text-sm mt-0.5">{specialty}</p>
        {subspecialty && <p className="text-zinc-400 text-sm mt-0.5">{subspecialty}</p>}

        <div className="mt-4 space-y-1.5">
          <p className="text-zinc-500 text-sm flex items-center gap-1.5">
            <span className="text-zinc-300">🏥</span>
            {hospital}
          </p>
          <p className="text-zinc-500 text-sm flex items-center gap-1.5">
            <span className="text-zinc-300">📍</span>
            {location}
          </p>
        </div>
      </div>

      {verified && (
        <span className="inline-flex items-center gap-1 mt-4 text-xs font-semibold text-emerald-600 bg-emerald-50 px-2.5 py-1 rounded-full w-fit">
          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
          Verified
        </span>
      )}

      <a
        href={`mailto:${email}?subject=Consultation Request via SkinAI`}
        className="mt-4 block w-full py-2.5 text-center border-2 border-pink-100 text-pink-800 rounded-xl font-semibold hover:bg-pink-50 hover:border-pink-200 transition-colors text-sm"
      >
        Request Consultation
      </a>
    </div>
  )
}
