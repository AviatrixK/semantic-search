export interface ClipboardLike {
  writeText?: (text: string) => Promise<void>
}

/** Copies text; false when the browser has no clipboard API (insecure origin, old browser) or refuses. */
export async function copyText(text: string, clipboard: ClipboardLike | undefined = globalThis.navigator?.clipboard): Promise<boolean> {
  if (!clipboard?.writeText) return false
  try {
    await clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
