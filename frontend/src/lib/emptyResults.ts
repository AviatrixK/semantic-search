export interface EmptyResults {
  title: string
  /** Set when videos are still being processed: their moments cannot be found yet. */
  processingNote: string | null
}

/** What to say when a search finds nothing. `processing` = videos not searchable yet (status "uploaded"). */
export function emptyResults(query: string, processing: number): EmptyResults {
  const n = Number.isFinite(processing) ? Math.max(0, Math.floor(processing)) : 0
  return {
    title: `No results for '${query}'`,
    processingNote:
      n === 0 ? null : `${n} video${n === 1 ? ' is' : 's are'} still processing and ${n === 1 ? 'is' : 'are'} not searchable yet.`,
  }
}
