// Each decision writes one JSON line to logs/gate-YYYY-MM-DD.jsonl
import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

export function makeLogger(logDir) {
  mkdirSync(logDir, { recursive: true })
  return function log(record) {
    const now = new Date()
    const file = join(logDir, `gate-${now.toISOString().slice(0, 10)}.jsonl`)
    try {
      appendFileSync(file, JSON.stringify({ time: now.toISOString(), ...record }) + '\n')
    } catch (err) {
      console.error('[safety-gate] failed to write log:', err)
    }
  }
}
