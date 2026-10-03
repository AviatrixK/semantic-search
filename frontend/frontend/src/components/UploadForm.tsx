import { useRef, useState } from 'react'
import type { DragEvent, FormEvent } from 'react'
import { formatBytes } from '../lib/format'
import { ALLOWED_EXTENSIONS, defaultTitle, validateVideoFile } from '../lib/uploadValidation'
import styles from './UploadForm.module.css'

interface Staged {
  key: string
  file: File
  title: string
}

interface Props {
  maxMb: number
  onSubmit: (entries: { file: File; title: string }[]) => void
}

const ACCEPT = ALLOWED_EXTENSIONS.map((e) => `.${e}`).join(',')
const keyOf = (f: File) => `${f.name}|${f.size}|${f.lastModified}`

/** Drag-and-drop zone and file picker. Chosen files are listed with an editable title before anything is uploaded. */
export default function UploadForm({ maxMb, onSubmit }: Props) {
  const [staged, setStaged] = useState<Staged[]>([])
  const [problems, setProblems] = useState<string[]>([])
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  function addFiles(files: File[]) {
    const rejected: string[] = []
    const accepted: Staged[] = []
    for (const file of files) {
      const problem = validateVideoFile(file, maxMb)
      if (problem) rejected.push(problem)
      else accepted.push({ key: keyOf(file), file, title: defaultTitle(file.name) })
    }
    setProblems(rejected)
    setStaged((cur) => [...cur, ...accepted.filter((a) => !cur.some((c) => c.key === a.key))])
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    addFiles(Array.from(e.dataTransfer.files))
  }

  function submit(e: FormEvent) {
    e.preventDefault()
    if (staged.length === 0) return
    onSubmit(staged.map(({ file, title }) => ({ file, title })))
    setStaged([])
    setProblems([])
  }

  return (
    <form className={styles.form} onSubmit={submit}>
      <label
        className={`${styles.drop} ${dragging ? styles.dragging : ''}`}
        onDragEnter={(e) => { e.preventDefault(); setDragging(true) }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false) }}
        onDrop={onDrop}
      >
        <input
          ref={input}
          className={styles.input}
          type="file"
          multiple
          accept={ACCEPT}
          onChange={(e) => {
            addFiles(Array.from(e.target.files ?? []))
            e.target.value = '' // lets the same file be chosen again after removing it
          }}
        />
        <strong>Drop videos here</strong>
        <span>or click to choose files</span>
        <small>MP4, WebM, MOV or MKV, up to {maxMb} MB each</small>
      </label>

      {problems.length > 0 && (
        <ul className={styles.problems} role="alert">
          {problems.map((p) => <li key={p}>{p}</li>)}
        </ul>
      )}

      {staged.length > 0 && (
        <>
          <ul className={styles.list}>
            {staged.map((s) => (
              <li key={s.key} className={styles.row}>
                <div className={styles.meta}>
                  <span className={styles.name} title={s.file.name}>{s.file.name}</span>
                  <span className={styles.size}>{formatBytes(s.file.size)}</span>
                </div>
                <input
                  className="input"
                  aria-label={`Title for ${s.file.name}`}
                  value={s.title}
                  maxLength={200}
                  placeholder="Title"
                  onChange={(e) => setStaged((cur) => cur.map((c) => (c.key === s.key ? { ...c, title: e.target.value } : c)))}
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  aria-label={`Remove ${s.file.name}`}
                  onClick={() => setStaged((cur) => cur.filter((c) => c.key !== s.key))}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
          <button type="submit" className="btn btn-primary">
            Upload {staged.length} video{staged.length === 1 ? '' : 's'}
          </button>
        </>
      )}
    </form>
  )
}
