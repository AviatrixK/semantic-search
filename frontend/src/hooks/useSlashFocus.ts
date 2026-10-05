import { useEffect } from 'react'
import type { RefObject } from 'react'

/** Pressing "/" anywhere on the page (outside a text field) focuses the given input, like on GitHub or YouTube. */
export function useSlashFocus(ref: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented || e.isComposing) return
      const t = e.target as HTMLElement | null
      if (t && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName))) return
      e.preventDefault()
      ref.current?.focus()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [ref])
}
