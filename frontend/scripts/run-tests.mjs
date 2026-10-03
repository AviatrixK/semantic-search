// Runs tests/*.test.mjs with Node's built-in test runner (no extra dependency). `npm test` compiles
// src/api and src/lib to .test-build first. Listing files explicitly keeps this working on every Node version
// and on Windows shells that do not expand globs.
import { readdirSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

const dir = fileURLToPath(new URL('../tests/', import.meta.url))
const files = readdirSync(dir).filter((f) => f.endsWith('.test.mjs')).map((f) => join(dir, f))
const result = spawnSync(process.execPath, ['--test', ...files], { stdio: 'inherit' })
process.exit(result.status ?? 1)
