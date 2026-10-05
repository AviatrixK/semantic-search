import { Suspense, lazy, useEffect } from 'react'
import { Outlet, Route, Routes, useLocation } from 'react-router-dom'
import Header from './components/Header'
import ProtectedRoute from './components/ProtectedRoute'
import RouteErrorBoundary from './components/RouteErrorBoundary'
import Sidebar from './components/Sidebar'
import Spinner from './components/Spinner'
import { useAuth } from './context/AuthContext'
import { useSidebar } from './hooks/useSidebar'
import { documentTitle, pageName } from './lib/titles'
import Home from './pages/Home'
import Library from './pages/Library'
import Login from './pages/Login'
import NotFound from './pages/NotFound'
import Register from './pages/Register'
import Search from './pages/Search'
import { loadAdmin, loadAsk, loadWatch } from './routes'
import styles from './App.module.css'

// Loaded on demand: the first screen (search, library, login) does not pay for the chat, the player page or the admin tools.
const Ask = lazy(loadAsk)
const Watch = lazy(loadWatch)
const AdminVideos = lazy(loadAdmin)

/** Holds back the whole app until we know whether the user is logged in (the refresh cookie check on load). */
function Layout() {
  const { user, status, bootstrapError, retryBootstrap } = useAuth()
  const { pathname } = useLocation()
  const sidebar = useSidebar()

  // The browser tab names the page, and a screen reader hears where a client-side navigation went.
  useEffect(() => {
    document.title = documentTitle(pathname)
  }, [pathname])

  if (status === 'loading') return <Spinner label="Loading…" />
  if (status === 'error') {
    return (
      <main className={styles.fatal}>
        <h1>Can't load the app</h1>
        <p role="alert">{bootstrapError}</p>
        <button type="button" className="btn btn-primary" onClick={retryBootstrap}>
          Try again
        </button>
      </main>
    )
  }
  return (
    <div className={styles.shell} data-collapsed={sidebar.collapsed ? '' : undefined} data-noside={user ? undefined : ''}>
      <a className="skip-link" href="#main">Skip to content</a>
      <Header onToggleSidebar={sidebar.toggle} />
      <Sidebar collapsed={sidebar.collapsed} />
      <div role="status" aria-live="polite" className="sr-only">{pageName(pathname)}</div>
      <div id="main" tabIndex={-1} className={styles.main}>
        <RouteErrorBoundary key={pathname}>
          <Suspense fallback={<Spinner label="Loading…" />}>
            <Outlet />
          </Suspense>
        </RouteErrorBoundary>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Home />
            </ProtectedRoute>
          }
        />
        <Route
          path="/search"
          element={
            <ProtectedRoute>
              <Search />
            </ProtectedRoute>
          }
        />
        <Route
          path="/ask"
          element={
            <ProtectedRoute>
              <Ask />
            </ProtectedRoute>
          }
        />
        <Route
          path="/library"
          element={
            <ProtectedRoute>
              <Library />
            </ProtectedRoute>
          }
        />
        <Route
          path="/watch/:videoId"
          element={
            <ProtectedRoute>
              <Watch />
            </ProtectedRoute>
          }
        />
        <Route
          path="/admin"
          element={
            <ProtectedRoute role="admin">
              <AdminVideos />
            </ProtectedRoute>
          }
        />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
