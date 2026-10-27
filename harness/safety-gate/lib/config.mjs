// Read config.json (in the same directory as gate.mjs). Another config file can be specified with the GATE_CONFIG environment variable.
import { readFileSync } from 'node:fs'
import { join, dirname, isAbsolute } from 'node:path'
import { fileURLToPath } from 'node:url'

export const PLUGIN_DIR = join(dirname(fileURLToPath(import.meta.url)), '..')

export function loadConfig() {
  const path = process.env.GATE_CONFIG || join(PLUGIN_DIR, 'config.json')
  const cfg = JSON.parse(readFileSync(path, 'utf-8'))
  // If log_dir is a relative path, it is relative to the plugin directory
  if (cfg.log_dir && !isAbsolute(cfg.log_dir)) cfg.log_dir = join(PLUGIN_DIR, cfg.log_dir)
  return cfg
}
