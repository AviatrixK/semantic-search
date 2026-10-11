import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { Video } from '../api'
import { describeError } from '../api/errors'

export function useVideos() {
  const [videos, setVideos] = useState<Video[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const latest = useRef(0)

  const load = useCallback(async (silent: boolean) => {
    const id = ++latest.current // only the newest request may update the screen
    if (!silent) setLoading(true)
    try {
      const list = await api.request<Video[]>('/api/videos')
      if (id !== latest.current) return
      setVideos(list)
      setError(null)
    } catch (e) {
      if (id === latest.current) setError(describeError(e))
    } finally {
      if (id === latest.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(false)
  }, [load])

  return {
    videos,
    loading,
    error,
    reload: useCallback(() => load(true), [load]),
    retry: useCallback(() => load(false), [load]),
  }
}
