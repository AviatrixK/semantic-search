import { useEffect, useState } from 'react'

/** Whole seconds left until `until` (a Date.now() timestamp), re-rendering every 250 ms (no network involved). 0 when null or past. */
export function useCountdown(until: number | null): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (until === null) return
    let id: ReturnType<typeof setTimeout>
    const tick = () => {
      setNow(Date.now())
      id = setTimeout(tick, 250)
    }
    tick()
    return () => clearTimeout(id)
  }, [until])
  return until === null ? 0 : Math.max(0, Math.ceil((until - now) / 1000))
}
