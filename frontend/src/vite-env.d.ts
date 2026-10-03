/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional API origin. Leave unset in dev: the Vite proxy makes /api and /auth same-origin. */
  readonly VITE_API_BASE?: string
}
