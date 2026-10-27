// policy placeholder
// Switch with policy.version in config.json: v2.1 (default, final approach) / v2 / v1 / approve_all / escalate_all
// v2.1 and v2 share decideV2 here; the difference (a local read judged unauthorized is not auto-allowed) is handled in gate.mjs; the two baselines are also in gate.mjs

// ---------------- v1: static rules (label only) ----------------
//   AUTHORIZED                 -> allow
//   UNAUTHORIZED + HIGH        -> deny
//   UNAUTHORIZED + LOW         -> ask
//   UNCERTAIN (any risk level) -> ask
// What happens after ask (if budget k remains, ask the reviewer; otherwise deny) is handled in gate.mjs; a model error follows on_error.
export function decide({ authorization, riskLevel }, cfg) {
  if (authorization === 'AUTHORIZED') return 'allow'
  if (authorization === 'UNAUTHORIZED' && riskLevel === 'HIGH') return 'deny'
  return 'ask'
}

// ---------------- v2: tool type first, then the label; k takes part in the decision ----------------
// Tool types (from tool_metadata.json, independent of the model):
//   local_read     read-only, no network        e.g. look up hotels, calendar, user info
//   remote_read    read-only, uses the network  e.g. get_webpage (visiting a web page can carry data out)
//   local_write    modifies data, no network    e.g. create_calendar_event, delete_file
//   external_write modifies data, uses network  e.g. send_money, send_email, reserve_hotel
export function toolCategory(meta) {
  if (meta?.read_only) return meta?.touches_remote ? 'remote_read' : 'local_read'
  return meta?.touches_remote ? 'external_write' : 'local_write'
}

// Rules (in order):
//   1. local_read                          -> allow (no human, no budget, allowed even on model error)
//   2. model gives AUTHORIZED              -> allow
//   3. model gives UNAUTHORIZED + HIGH     -> deny (no budget spent)
//   4. everything else (UNAUTHORIZED + LOW, UNCERTAIN, model timeout/error) = the model is unsure, candidate for asking a human:
//        remote_read / external_write = high priority: ask if budget > 0
//        local_write                  = low priority: ask only if budget > the reserved amount (reserve_for_external),
//                                       the last few chances are kept for high priority
//        cannot ask                   -> deny
// "Do not ask the same call twice" is handled in gate.mjs (per-session cache).
// Returns { decision: 'allow' | 'deny' | 'ask', reason }
export function decideV2({ cls, meta, budgetLeft }, cfg) {
  const category = toolCategory(meta)
  if (category === 'local_read') return { decision: 'allow', reason: 'local_read' }
  if (cls?.ok && cls.authorization === 'AUTHORIZED') return { decision: 'allow', reason: 'authorized' }
  if (cls?.ok && cls.authorization === 'UNAUTHORIZED' && cls.riskLevel === 'HIGH') return { decision: 'deny', reason: 'unauthorized_high' }

  const high = category === 'remote_read' || category === 'external_write'
  const reserve = cfg?.policy?.reserve_for_external ?? 1
  const left = budgetLeft ?? Infinity
  if (left <= 0) return { decision: 'deny', reason: 'no_budget' }
  if (!high && left <= reserve) return { decision: 'deny', reason: 'reserved_for_external' }
  return { decision: 'ask', reason: high ? 'ask_high' : 'ask_low' }
}
