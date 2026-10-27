// Tool lookup table: tool_metadata.json
// A tool with the same name is annotated the same way in every suite, so it is looked up directly by tool name.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PLUGIN_DIR } from './config.mjs'

const table = {}
const raw = JSON.parse(readFileSync(join(PLUGIN_DIR, 'tool_metadata.json'), 'utf-8'))
for (const suite of Object.values(raw)) {
  for (const [name, meta] of Object.entries(suite)) {
    table[name] = { read_only: meta.read_only, touches_remote: meta.touches_remote }
  }
}

// Returns { meta, found }. A tool not in the table is treated for now as "not read-only, affects the outside", and is marked in the log.
export function lookupMetadata(toolName) {
  if (Object.hasOwn(table, toolName)) return { meta: table[toolName], found: true }
  return { meta: { read_only: false, touches_remote: true }, found: false }
}
