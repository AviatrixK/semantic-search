/** Mirrors the backend policy (app/schemas/auth.py) so users see problems before submitting. The server stays the authority. */
export interface PasswordCheck {
  label: string
  ok: boolean
}

export function checkPassword(password: string): PasswordCheck[] {
  return [
    { label: 'At least 8 characters', ok: password.length >= 8 },
    { label: 'Contains a letter', ok: /\p{L}/u.test(password) },
    { label: 'Contains a digit', ok: /\p{N}/u.test(password) },
    { label: 'At most 72 bytes', ok: new TextEncoder().encode(password).length <= 72 },
  ]
}

export function passwordIsValid(password: string): boolean {
  return checkPassword(password).every((c) => c.ok)
}
