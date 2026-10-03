import { createApiClient, webLocksRunner } from './client'

/** The app-wide client. The access token lives only in this module's memory (never localStorage). */
export const api = createApiClient({
  baseUrl: import.meta.env.VITE_API_BASE ?? '',
  runExclusive: webLocksRunner(),
})

export interface User {
  id: string
  email: string
  role: 'user' | 'admin'
}

export interface TokenOut {
  access_token: string
  token_type: string
}
