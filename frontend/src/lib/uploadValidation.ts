import { formatBytes } from './format.js'

/** Same allow-list as the backend (app/api/routes/videos.py). The server stays the authority; this gives instant feedback. */
export const ALLOWED_EXTENSIONS = ['mp4', 'webm', 'mov', 'mkv'] as const

const MIME_BY_EXTENSION: Record<string, string> = {
  mp4: 'video/mp4',
  webm: 'video/webm',
  mov: 'video/quicktime',
  mkv: 'video/x-matroska',
}

export interface FileLike {
  name: string
  size: number
}

export function extensionOf(filename: string): string {
  const i = filename.lastIndexOf('.')
  return i < 0 ? '' : filename.slice(i + 1).toLowerCase()
}

/** The MIME type the backend expects for this extension. Browsers often report "" for .mkv, which the server would reject. */
export function mimeForFile(filename: string): string | undefined {
  return MIME_BY_EXTENSION[extensionOf(filename)]
}

/** "My Talk.mp4" -> "My Talk" */
export function defaultTitle(filename: string): string {
  const i = filename.lastIndexOf('.')
  return (i > 0 ? filename.slice(0, i) : filename).trim() || filename
}

/** Returns a user-facing problem, or null if the file looks uploadable. */
export function validateVideoFile(file: FileLike, maxMb: number): string | null {
  if (mimeForFile(file.name) === undefined) {
    return `"${file.name}" is not a supported video type. Use MP4, WebM, MOV or MKV.`
  }
  if (file.size === 0) return `"${file.name}" is empty.`
  if (file.size > maxMb * 1024 * 1024) {
    return `"${file.name}" is ${formatBytes(file.size)}, over the ${maxMb} MB limit.`
  }
  return null
}

/**
 * The multipart body for POST /api/videos. The file is re-wrapped with the MIME type the backend expects, because
 * browsers often report "" for .mkv (the server would answer 415). Wrapping a Blob copies no bytes.
 */
export function buildUploadForm(file: File, title: string): FormData {
  const form = new FormData()
  form.append('file', new File([file], file.name, { type: mimeForFile(file.name) ?? file.type }))
  if (title.trim()) form.append('title', title.trim())
  return form
}
