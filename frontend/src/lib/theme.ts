/** The app is light by default. Dark is an explicit choice (it is never switched on by the operating system's setting). */
export type ThemeChoice = 'light' | 'dark'

export const THEME_STORAGE_KEY = 'svs.theme'

/** Anything but "dark" (nothing saved, corrupt storage, an old "system" value) is light. */
export function parseTheme(value: unknown): ThemeChoice {
  return value === 'dark' ? 'dark' : 'light'
}

export function nextTheme(choice: ThemeChoice): ThemeChoice {
  return choice === 'light' ? 'dark' : 'light'
}

export interface RootLike {
  setAttribute(name: string, value: string): void
  removeAttribute(name: string): void
}

export function applyTheme(choice: ThemeChoice, root: RootLike): void {
  root.setAttribute('data-theme', choice)
}

const LABEL: Record<ThemeChoice, string> = { light: 'Light', dark: 'Dark' }

/** Text for the toggle: what is active now and what a click switches to. */
export function themeText(choice: ThemeChoice): { current: string; next: string; label: string } {
  const next = nextTheme(choice)
  return { current: LABEL[choice], next: LABEL[next], label: `Theme: ${LABEL[choice]}. Switch to ${LABEL[next].toLowerCase()}.` }
}
