import { useCallback, useState } from 'react'

const KEY = 'svs.sidebar'

/** Whether the left menu is collapsed to icons. Remembered; a blocked storage just means it is not remembered. */
export function useSidebar() {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(KEY) === 'collapsed'
    } catch {
      return false
    }
  })
  const toggle = useCallback(() => {
    setCollapsed((c) => {
      try {
        localStorage.setItem(KEY, c ? 'expanded' : 'collapsed')
      } catch {
        /* ignore */
      }
      return !c
    })
  }, [])
  return { collapsed, toggle }
}
