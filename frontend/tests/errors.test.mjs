import test from 'node:test'
import assert from 'node:assert/strict'
import {
  ApiError, NetworkError, describeError, detailMessage, formatWait, parseRetryAfter, rateLimitMessage,
} from '../.test-build/api/errors.js'
import { checkPassword, passwordIsValid } from '../.test-build/lib/password.js'

test('parseRetryAfter: seconds, HTTP dates, and junk', () => {
  assert.equal(parseRetryAfter('42'), 42)
  assert.equal(parseRetryAfter(' 7 '), 7)
  assert.equal(parseRetryAfter('1.2'), 2)
  assert.equal(parseRetryAfter('0'), 0)
  const now = Date.parse('2026-01-01T00:00:00Z')
  assert.equal(parseRetryAfter('Thu, 01 Jan 2026 00:00:30 GMT', now), 30)
  assert.equal(parseRetryAfter('Wed, 31 Dec 2025 23:00:00 GMT', now), 0) // date in the past
  assert.equal(parseRetryAfter('soon'), undefined)
  assert.equal(parseRetryAfter(''), undefined)
  assert.equal(parseRetryAfter(null), undefined)
})

test('formatWait', () => {
  assert.equal(formatWait(1), '1 second')
  assert.equal(formatWait(45), '45 seconds')
  assert.equal(formatWait(0), '1 second') // never "0 seconds"
  assert.equal(formatWait(60), '1 minute')
  assert.equal(formatWait(65), '1 minute 5 seconds')
  assert.equal(formatWait(120), '2 minutes')
})

test('detailMessage: FastAPI string and 422 list formats', () => {
  assert.equal(detailMessage({ detail: 'Email already registered' }), 'Email already registered')
  assert.equal(
    detailMessage({ detail: [{ msg: 'Value error, Password must contain at least one letter and one digit' }] }),
    'Password must contain at least one letter and one digit',
  )
  assert.equal(detailMessage({ detail: [{ msg: 'a' }, { msg: 'b' }] }), 'a b')
  assert.equal(detailMessage({}), undefined)
  assert.equal(detailMessage(null), undefined)
  assert.equal(detailMessage({ detail: 5 }), undefined)
})

test('describeError: 429 uses Retry-After and says how long to wait', () => {
  assert.equal(describeError(new ApiError(429, 'x', 42), 'login'), 'Too many attempts. Please wait 42 seconds and try again.')
  assert.equal(describeError(new ApiError(429, 'x', 75)), 'Too many attempts. Please wait 1 minute 15 seconds and try again.')
  assert.equal(describeError(new ApiError(429, 'x')), 'Too many attempts. Please wait a moment and try again.')
  assert.equal(rateLimitMessage(3), 'Too many attempts. Please wait 3 seconds and try again.')
})

test('describeError: other statuses and contexts', () => {
  assert.equal(describeError(new ApiError(401, 'Invalid credentials'), 'login'), 'Incorrect email or password.')
  assert.match(describeError(new ApiError(401, 'x')), /session has expired/)
  assert.equal(describeError(new ApiError(409, 'Email already registered'), 'register'), 'An account with this email already exists.')
  assert.equal(describeError(new ApiError(422, 'Password must contain at least one letter and one digit')),
    'Password must contain at least one letter and one digit')
  assert.match(describeError(new ApiError(500, 'boom')), /went wrong on the server/)
  assert.doesNotMatch(describeError(new ApiError(500, 'Traceback secret')), /secret/)
  assert.match(describeError(new NetworkError()), /Can't reach the server/)
  assert.equal(describeError('weird'), 'Something went wrong. Please try again.')
})

test('password rules mirror the backend policy', () => {
  assert.equal(passwordIsValid('Passw0rd123'), true)
  assert.equal(passwordIsValid('short1'), false)
  assert.equal(passwordIsValid('allletters'), false)
  assert.equal(passwordIsValid('12345678'), false)
  assert.equal(passwordIsValid('a1'.repeat(40)), false) // 80 bytes > 72
})

test('password rules: bytes not characters, and which rule fails', () => {
  assert.equal(passwordIsValid('é1'.repeat(24)), true) // 72 bytes exactly
  assert.equal(passwordIsValid('é1'.repeat(25)), false) // 75 bytes
  const failed = checkPassword('abcdefgh').filter((c) => !c.ok).map((c) => c.label)
  assert.deepEqual(failed, ['Contains a digit'])
})
