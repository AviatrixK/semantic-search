/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional API origin. Leave unset in dev: the Vite proxy makes /api and /auth same-origin. */
  readonly VITE_API_BASE?: string
  /** Client-side upload size check. Keep equal to the backend's MAX_UPLOAD_MB (default 500). */
  readonly VITE_MAX_UPLOAD_MB?: string
}
