export const APP_NAME = 'Semantic Video Search'

/** The page name for a path: used for the browser tab title and the screen-reader announcement on navigation. */
export function pageName(pathname: string): string {
  const path = pathname.replace(/\/+$/, '') || '/'
  if (path === '/') return 'Home'
  if (path === '/search') return 'Search results'
  if (path === '/ask') return 'Ask'
  if (path === '/library') return 'Library'
  if (path === '/admin') return 'Admin'
  if (path === '/login') return 'Log in'
  if (path === '/register') return 'Register'
  if (path === '/forbidden') return 'Not allowed'
  if (/^\/watch(\/|$)/.test(path)) return 'Watch'
  return 'Page not found'
}

export function documentTitle(pathname: string): string {
  return `${pageName(pathname)} · ${APP_NAME}`
}
