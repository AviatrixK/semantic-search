import { Outlet, Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar'
import ProtectedRoute from './components/ProtectedRoute'
import Spinner from './components/Spinner'
import { useAuth } from './context/AuthContext'
import Admin from './pages/Admin'
import Home from './pages/Home'
import Login from './pages/Login'
import NotFound from './pages/NotFound'
import Register from './pages/Register'
import styles from './App.module.css'

/** Holds back the whole app until we know whether the user is logged in (the refresh cookie check on load). */
function Layout() {
  const { status, bootstrapError, retryBootstrap } = useAuth()

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
    <>
      <Navbar />
      <Outlet />
    </>
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
          path="/admin"
          element={
            <ProtectedRoute role="admin">
              <Admin />
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
