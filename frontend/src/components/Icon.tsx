import type { ReactNode } from 'react'

const PATHS: Record<string, ReactNode> = {
  home: <path d="M4 11.5 12 4l8 7.5V20a1 1 0 0 1-1 1h-4.5v-6h-5v6H5a1 1 0 0 1-1-1z" />,
  ask: <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z" />,
  library: <path d="M5 4h4v16H5zM11 4h4v16h-4zM17.2 5.3l3.6-1 3 14.6-3.6 1z" transform="translate(-1.5 0)" />,
  admin: <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zM19 12l2-1.2-2-3.5-2.3.8a7 7 0 0 0-1.6-.9L14.7 4h-4l-.4 2.2a7 7 0 0 0-1.6.9l-2.3-.8-2 3.5L6 12a7 7 0 0 0 0 1.9l-1.9 1.3 2 3.5 2.3-.8c.5.4 1 .7 1.6.9l.4 2.2h4l.4-2.2c.6-.2 1.1-.5 1.6-.9l2.3.8 2-3.5z" />,
  search: <path d="M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zM15.5 15.5 21 21" />,
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  sun: <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z" />,
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />,
  logout: <path d="M10 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h5M15 8l4 4-4 4M19 12H9" />,
  cap: <path d="M2.5 9.5 12 5l9.5 4.5L12 14zM6 11.8V16c3.4 2.3 8.6 2.3 12 0v-4.2M21.5 9.5V15" />,
  book: <path d="M4 5.5c3-1.4 5.6-1 8 .8 2.4-1.8 5-2.2 8-.8v13c-3-1.4-5.6-1-8 .8-2.4-1.8-5-2.2-8-.8zM12 6.3v13" />,
}

/** Small line icons drawn inline (no icon font, no images). Decorative: the text next to them carries the meaning. */
export default function Icon({ name, size = 24 }: { name: keyof typeof PATHS; size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {PATHS[name]}
    </svg>
  )
}
