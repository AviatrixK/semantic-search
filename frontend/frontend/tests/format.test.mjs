import test from 'node:test'
import assert from 'node:assert/strict'
import { formatBytes, formatDate, formatDuration, isTerminalStage, stageLabel, statusBadge } from '../.test-build/lib/format.js'
import { defaultTitle, extensionOf, mimeForFile, validateVideoFile } from '../.test-build/lib/uploadValidation.js'

test('formatDuration is mm:ss with unbounded minutes', () => {
  assert.equal(formatDuration(0), '00:00')
  assert.equal(formatDuration(59), '00:59')
  assert.equal(formatDuration(61), '01:01')
  assert.equal(formatDuration(1700), '28:20')
  assert.equal(formatDuration(3912), '65:12')
  assert.equal(formatDuration(59.9), '00:59')
})

test('formatDuration: unknown or invalid is an em dash', () => {
  for (const v of [null, undefined, NaN, -1, Infinity]) assert.equal(formatDuration(v), '—')
})

test('formatBytes', () => {
  assert.equal(formatBytes(0), '0 B')
  assert.equal(formatBytes(1023), '1023 B')
  assert.equal(formatBytes(1536), '1.5 KB')
  assert.equal(formatBytes(5 * 1024 * 1024), '5.0 MB')
  assert.equal(formatBytes(123 * 1024 * 1024), '123 MB')
  assert.equal(formatBytes(2.5 * 1024 ** 3), '2.5 GB')
  assert.equal(formatBytes(-1), '—')
})

test('formatDate: readable, and an em dash for garbage', () => {
  const s = formatDate('2026-03-04T05:06:07Z', 'en-US', 'UTC')
  assert.match(s, /Mar/)
  assert.match(s, /2026/)
  assert.match(s, /05:06/)
  assert.equal(formatDate('not a date'), '—')
})

test('stage labels and terminal stages', () => {
  assert.equal(stageLabel('queued'), 'Waiting in queue')
  assert.equal(stageLabel('transcribing'), 'Transcribing speech')
  assert.equal(stageLabel('mystery'), 'mystery')
  assert.equal(isTerminalStage('done'), true)
  assert.equal(isTerminalStage('failed'), true)
  assert.equal(isTerminalStage('embedding'), false)
})

test('status badges: "uploaded" means processing', () => {
  assert.deepEqual(statusBadge('ready'), { label: 'Ready', tone: 'ok' })
  assert.deepEqual(statusBadge('uploaded'), { label: 'Processing', tone: 'busy' })
  assert.deepEqual(statusBadge('failed'), { label: 'Failed', tone: 'bad' })
  assert.deepEqual(statusBadge('weird'), { label: 'weird', tone: 'neutral' })
})

// ---- upload validation
test('extensionOf, mimeForFile and defaultTitle', () => {
  assert.equal(extensionOf('Talk.MP4'), 'mp4')
  assert.equal(extensionOf('noext'), '')
  assert.equal(extensionOf('a.b.mkv'), 'mkv')
  assert.equal(mimeForFile('clip.mkv'), 'video/x-matroska') // browsers often send "" for .mkv
  assert.equal(mimeForFile('clip.MOV'), 'video/quicktime')
  assert.equal(mimeForFile('clip.exe'), undefined)
  assert.equal(defaultTitle('My Talk.mp4'), 'My Talk')
  assert.equal(defaultTitle('a.b.mp4'), 'a.b')
  assert.equal(defaultTitle('.hidden'), '.hidden')
})

test('validateVideoFile', () => {
  const MB = 1024 * 1024
  assert.equal(validateVideoFile({ name: 'a.mp4', size: 10 * MB }, 500), null)
  assert.equal(validateVideoFile({ name: 'A.WEBM', size: 500 * MB }, 500), null) // exactly at the limit
  assert.match(validateVideoFile({ name: 'notes.txt', size: 5 }, 500), /not a supported video type/)
  assert.match(validateVideoFile({ name: 'movie', size: 5 }, 500), /not a supported video type/)
  assert.match(validateVideoFile({ name: 'a.mp4.exe', size: 5 }, 500), /not a supported video type/)
  assert.match(validateVideoFile({ name: 'a.mp4', size: 0 }, 500), /is empty/)
  const big = validateVideoFile({ name: 'big.mp4', size: 501 * MB }, 500)
  assert.match(big, /501 MB/)
  assert.match(big, /500 MB limit/)
})

test('buildUploadForm: re-wraps the file with the right MIME type (browsers send "" for .mkv) and trims the title', async () => {
  const { buildUploadForm } = await import('../.test-build/lib/uploadValidation.js')
  const mkv = new File([new Uint8Array(10)], 'Talk.mkv', { type: '' })
  const form = buildUploadForm(mkv, '  My Talk  ')
  const sent = form.get('file')
  assert.equal(sent.name, 'Talk.mkv')
  assert.equal(sent.type, 'video/x-matroska')
  assert.equal(sent.size, 10)
  assert.equal(form.get('title'), 'My Talk')
  assert.equal(buildUploadForm(new File([new Uint8Array(1)], 'a.mp4', { type: 'video/mp4' }), '   ').has('title'), false)
  assert.equal(buildUploadForm(new File([new Uint8Array(1)], 'a.MOV', { type: 'application/octet-stream' }), '').get('file').type, 'video/quicktime')
})
