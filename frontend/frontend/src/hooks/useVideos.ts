import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { Video } from '../api'
import { describeError } from '../api/errors'

const REFRESH_WHILE_PROCESSING_MS = 5000

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

  // Videos still processing (e.g. after a page reload, when the job cards are gone): keep the table fresh.
  const anyProcessing = videos?.some((v) => v.status === 'uploaded') ?? false
  useEffect(() => {
    if (!anyProcessing) return
    const id = setInterval(() => void load(true), REFRESH_WHILE_PROCESSING_MS)
    return () => clearInterval(id)
  }, [anyProcessing, load])

  return {
    videos,
    loading,
    error,
    reload: useCallback(() => load(true), [load]),
    retry: useCallback(() => load(false), [load]),
  }
}
