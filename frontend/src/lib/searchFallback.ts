/** The search endpoint sets `X-Search-Fallback: keyword-only` when the embedding service was down or too slow and it answered from the
 * keyword index alone (see backend/app/api/routes/search.py). The page tells the user, above the results. */
export const KEYWORD_ONLY_NOTE = 'Showing keyword matches only, as smart search is temporarily unavailable.'

export function isKeywordOnly(headers: { get(name: string): string | null } | null | undefined): boolean {
  return (headers?.get('X-Search-Fallback') ?? '').trim().toLowerCase() === 'keyword-only'
}
