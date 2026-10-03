import { useEffect, useRef } from 'react'
import styles from './ConfirmDialog.module.css'

interface Props {
  open: boolean
  title: string
  message: string
  confirmLabel: string
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
}

/** Modal confirmation on the native <dialog> element: focus trapping, Escape and the backdrop come from the browser. */
export default function ConfirmDialog({ open, title, message, confirmLabel, busy = false, onConfirm, onCancel }: Props) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])

  return (
    <dialog ref={ref} className={styles.dialog} onCancel={(e) => { e.preventDefault(); if (!busy) onCancel() }}
      aria-labelledby="confirm-title">
      <h2 id="confirm-title" className={styles.title}>{title}</h2>
      <p className={styles.message}>{message}</p>
      <div className={styles.actions}>
        <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={busy} autoFocus>Cancel</button>
        <button type="button" className={`btn ${styles.danger}`} onClick={onConfirm} disabled={busy}>
          {busy ? 'Working…' : confirmLabel}
        </button>
      </div>
    </dialog>
  )
}
