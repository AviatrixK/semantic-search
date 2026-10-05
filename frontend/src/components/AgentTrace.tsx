import { formatArgs, formatLatency, groupByStep, stepLine } from '../lib/trace'
import type { TraceStep } from '../lib/trace'
import styles from './AgentTrace.module.css'

interface Props {
  steps: TraceStep[]
  /** True while the answer is still being worked out: the list is shown open and grows live. */
  running: boolean
}

function Icon({ status }: { status: TraceStep['status'] }) {
  if (status === 'running') return <span className={styles.spinner} aria-label="running" role="img" />
  return (
    <span className={`${styles.icon} ${status === 'error' ? styles.bad : styles.good}`} role="img" aria-label={status === 'error' ? 'failed' : 'done'}>
      {status === 'error' ? '!' : '✓'}
    </span>
  )
}

function Row({ s }: { s: TraceStep }) {
  return (
    <li className={styles.row} data-status={s.status}>
      <details>
        <summary className={styles.line}>
          <Icon status={s.status} />
          <span className={styles.text}>{stepLine(s)}</span>
          {s.latencyMs !== null && <span className={styles.latency}>{formatLatency(s.latencyMs)}</span>}
        </summary>
        <dl className={styles.details}>
          <dt>Tool</dt>
          <dd><code>{s.tool}</code></dd>
          <dt>Arguments</dt>
          <dd><pre>{formatArgs(s.args)}</pre></dd>
          {s.summary && (
            <>
              <dt>Result</dt>
              <dd>{s.summary}</dd>
            </>
          )}
        </dl>
      </details>
    </li>
  )
}

/** The agent's work as a list: "Searching “x”…" while a call runs, "Found 5 clips" when it is done, details one click away. */
export default function AgentTrace({ steps, running }: Props) {
  if (steps.length === 0) return null
  const list = (
    <ol className={styles.rounds}>
      {groupByStep(steps).map((group) => (
        <li key={group[0].step} className={styles.round}>
          {group.length > 1 && <span className={styles.together}>{group.length} at the same time</span>}
          <ul className={styles.calls}>
            {group.map((s) => <Row key={`${s.step}.${s.index}`} s={s} />)}
          </ul>
        </li>
      ))}
    </ol>
  )
  if (running) return <div className={styles.live} role="group" aria-label="Research steps">{list}</div>
  return (
    <details className={styles.finished}>
      <summary className={styles.title}>How this was found ({steps.length} step{steps.length === 1 ? '' : 's'})</summary>
      {list}
    </details>
  )
}
