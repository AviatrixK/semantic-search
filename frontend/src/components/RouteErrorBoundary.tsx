import { Component } from 'react'
import type { ReactNode } from 'react'

interface State {
  failed: boolean
}

/** Catches a page that failed to load (a lazy chunk the network dropped, or a stale deploy) instead of leaving a blank screen. */
export default class RouteErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { failed: false }

  static getDerivedStateFromError(): State {
    return { failed: true }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <main style={{ maxWidth: '28rem', margin: '4rem auto', padding: '0 1rem', textAlign: 'center' }}>
        <h1>This page could not load</h1>
        <p style={{ color: 'var(--muted)' }}>Check your connection and try again. If it keeps happening, reload the page.</p>
        <button type="button" className="btn btn-primary" onClick={() => window.location.reload()}>Reload</button>
      </main>
    )
  }
}
