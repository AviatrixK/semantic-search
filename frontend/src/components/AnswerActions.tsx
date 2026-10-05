import { useEffect, useRef, useState } from 'react'
import type { Citation } from '../api'
import { answerAsText } from '../lib/answerText'
import { copyText } from '../lib/clipboard'
import styles from './AnswerActions.module.css'

interface Props {
  text: string
  citations: Citation[]
  /** False while another question is being answered. */
  canRegenerate: boolean
  onRegenerate: () => void
}

/** Copy the answer (with its sources) and ask the same question again. */
export default function AnswerActions({ text, citations, canRegenerate, onRegenerate }: Props) {
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle')
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])

  async function copy() {
    const ok = await copyText(answerAsText(text, citations))
    setCopied(ok ? 'copied' : 'failed')
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setCopied('idle'), 2500)
  }

  return (
    <div className={styles.row}>
      <button type="button" className={`btn btn-secondary ${styles.action}`} onClick={() => void copy()}>
        Copy answer
      </button>
      <button type="button" className={`btn btn-secondary ${styles.action}`} onClick={onRegenerate} disabled={!canRegenerate}
        title="Ask this question again and replace the answer">
        Regenerate
      </button>
      <span className={styles.status} role="status">
        {copied === 'copied' && 'Copied to the clipboard'}
        {copied === 'failed' && 'Could not copy: select the text instead'}
      </span>
    </div>
  )
}
