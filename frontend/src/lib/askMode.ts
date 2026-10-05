/** How the Ask page answers: let the server choose, or force one approach. Values are what GET /api/ask/stream accepts. */
export const ASK_MODES = ['auto', 'rag', 'agent'] as const
export type AskMode = (typeof ASK_MODES)[number]

export const ASK_MODE_STORAGE_KEY = 'svs.ask.mode'

export const ASK_MODE_OPTIONS: ReadonlyArray<{ value: AskMode; label: string; hint: string }> = [
  { value: 'auto', label: 'Automatic', hint: 'Picks the lightest approach that fits the question.' },
  { value: 'rag', label: 'Quick', hint: 'One search, one answer. Fast.' },
  { value: 'agent', label: 'Research', hint: 'Plans several searches and reads transcripts. Slower, for comparisons and multi-part questions.' },
]

export function parseAskMode(value: unknown): AskMode {
  return ASK_MODES.includes(value as AskMode) ? (value as AskMode) : 'auto'
}
