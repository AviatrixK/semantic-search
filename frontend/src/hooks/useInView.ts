import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'

/** True once the element has come near the screen (and stays true): lets a long page load thumbnails only when they are about to show. */
export function useInView<T extends Element>(rootMargin = '300px'): [RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null)
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === 'undefined')

  useEffect(() => {
    const el = ref.current
    if (seen || !el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true)
          observer.disconnect()
        }
      },
      { rootMargin },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [seen, rootMargin])

  return [ref, seen]
}
