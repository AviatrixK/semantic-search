/**
 * Languages offered on upload. A copy of backend/app/services/languages.py (tests/test_languages.py on the backend fails if the two
 * differ). The codes are Sarvam's; the server turns them into Whisper's where needed. Leaving the choice empty = detect automatically.
 */
export interface LanguageOption {
  code: string
  label: string
}

export const AUTO_LANGUAGE = ''

export const LANGUAGES: readonly LanguageOption[] = [
  { code: 'hi-IN', label: 'Hindi' },
  { code: 'en-IN', label: 'English' },
  { code: 'bn-IN', label: 'Bengali' },
  { code: 'ta-IN', label: 'Tamil' },
  { code: 'te-IN', label: 'Telugu' },
  { code: 'kn-IN', label: 'Kannada' },
  { code: 'ml-IN', label: 'Malayalam' },
  { code: 'mr-IN', label: 'Marathi' },
  { code: 'gu-IN', label: 'Gujarati' },
  { code: 'pa-IN', label: 'Punjabi' },
  { code: 'od-IN', label: 'Odia' },
  { code: 'as-IN', label: 'Assamese' },
  { code: 'ur-IN', label: 'Urdu' },
  { code: 'ne-IN', label: 'Nepali' },
  { code: 'kok-IN', label: 'Konkani' },
  { code: 'ks-IN', label: 'Kashmiri' },
  { code: 'sd-IN', label: 'Sindhi' },
  { code: 'sa-IN', label: 'Sanskrit' },
  { code: 'sat-IN', label: 'Santali' },
  { code: 'mni-IN', label: 'Manipuri' },
  { code: 'brx-IN', label: 'Bodo' },
  { code: 'mai-IN', label: 'Maithili' },
  { code: 'doi-IN', label: 'Dogri' },
]

/** True for "" (automatic) and for a code in the list: anything else must never be sent to the server. */
export function isLanguageChoice(value: unknown): value is string {
  return value === AUTO_LANGUAGE || (typeof value === 'string' && LANGUAGES.some((l) => l.code === value))
}
