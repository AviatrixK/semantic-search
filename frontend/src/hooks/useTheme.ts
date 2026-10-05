import { useCallback, useEffect, useState } from 'react'
import { THEME_STORAGE_KEY, applyTheme, nextTheme, parseTheme } from '../lib/theme'
import type { ThemeChoice } from '../lib/theme'

function stored(): ThemeChoice {
  try {
    return parseTheme(localStorage.getItem(THEME_STORAGE_KEY))
  } catch {
    return 'light' // storage blocked
  }
}

/** Light or dark, saved in localStorage and kept in step across tabs. index.html applies it before first paint. */
export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(stored)

  useEffect(() => applyTheme(choice, document.documentElement), [choice])

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === THEME_STORAGE_KEY || e.key === null) setChoice(parseTheme(e.newValue))
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  const cycle = useCallback(() => {
    const next = nextTheme(choice)
    setChoice(next)
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next)
    } catch {
      /* not saved, but it applies for this visit */
    }
  }, [choice])

  return { choice, cycle }
}
