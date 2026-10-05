// Pages that are not needed for the first screen are loaded on demand (smaller first download). Navbar links start the download
// as soon as the pointer or keyboard focus reaches them, so the click usually finds the page already there.
export const loadAsk = () => import('./pages/Ask')
export const loadWatch = () => import('./pages/Watch')
export const loadAdmin = () => import('./pages/AdminVideos')

const preloaders: Record<string, () => Promise<unknown>> = { '/ask': loadAsk, '/admin': loadAdmin }

export function preloadRoute(path: string): void {
  preloaders[path]?.().catch(() => undefined) // a failed prefetch is not an error: the real navigation reports it
}
