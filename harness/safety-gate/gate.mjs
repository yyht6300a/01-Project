// Safety approval gate plugin
// Before each tool call is executed:
//   1. Get data from dsh (tool name, arguments, user task, trace, session ID)
//   2. Look up the table to fill in tool_metadata
//   3. Call the safety gate model to get authorization + risk_level
//   4. Pick one of three with fixed thresholds: allow / deny / ask (we will work out the details later; I wrote it this way to keep the pipeline complete)
//      The default is now policy v2.1 (tool type first, then the label; k takes part in the decision), see lib/policy.mjs
//      v2.1 = v2 + a local read is not auto-allowed when the model explicitly judges it UNAUTHORIZED (in the offline replay the block rate went from 69% -> 95%)
//      v1, v2 and the two baselines (approve_all / escalate_all) are kept for comparison; switch with policy.version in config.json
//   5. ask: if budget > 0, hand it to the simulated reviewer (budget -1); if budget = 0, deny directly
//   6. Write the log and return the result to dsh

import { loadConfig } from './lib/config.mjs'
import { lookupMetadata } from './lib/metadata.mjs'
import { extract } from './lib/extract.mjs'
import { classify } from './lib/classifier.mjs'
import { decide, decideV2, toolCategory } from './lib/policy.mjs'
import { Budget } from './lib/budget.mjs'
import { review } from './lib/reviewer.mjs'
import { makeLogger } from './lib/logger.mjs'
import { DENY_MESSAGE, NOT_AVAILABLE_MESSAGE } from './lib/deny.mjs'

export const name = 'safety-gate'

// Serialize the argument object after sorting its keys, so the same arguments give the same string (used to decide "the same call")
function stableStringify(v) {
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(',')}]`
  if (v && typeof v === 'object') return `{${Object.keys(v).sort().map(k => `${JSON.stringify(k)}:${stableStringify(v[k])}`).join(',')}}`
  return JSON.stringify(v)
}

export function apply(ctx) {
  const cfg = loadConfig()
  const budget = new Budget(cfg.k)
  const log = makeLogger(cfg.log_dir)
  const reviewerDesc = cfg.reviewer?.mode === 'llm' ? `LLM (${cfg.reviewer.model})` : `placeholder (always ${cfg.reviewer?.placeholder_decision === 'approve' ? 'approve' : 'deny'})`
  const policyVersion = cfg.policy?.version ?? 'v2.1'
  // v2: an identical call (tool + arguments) in the same session is not asked again; reuse the previous final result directly
  const decided = new Map()
  console.log(`[safety-gate] loaded. policy=${policyVersion}, k=${cfg.k}, non-AgentDojo tools=${cfg.other_tools ?? 'deny'}, model service=${cfg.classifier.url}, reviewer=${reviewerDesc}`)

  async function callClassifier(info, meta, timeoutMs) {
    const c = timeoutMs ? { ...cfg, classifier: { ...cfg.classifier, timeout_ms: timeoutMs } } : cfg
    return classify({
      userTask: info.task,
      history: info.history,
      proposedToolCall: { function: info.toolName, args: info.toolArgs },
      toolMetadata: meta,
    }, c)
  }

  function recordClassifier(record, cls) {
    Object.assign(record, {
      authorization: cls.authorization,
      risk_level: cls.riskLevel,
      risk_factors: cls.riskFactors,
      rationale: cls.rationale,
      classifier_ok: cls.ok,
      classifier_error: cls.error,
      classifier_usage: cls.usage,
      classifier_latency_ms: cls.latencyMs,
    })
  }

  // Full policy v2 flow, returns 'allow' | 'deny'; all intermediate results are written into record
  async function decideWithV2(info, meta, record) {
    const category = toolCategory(meta)
    Object.assign(record, { policy_version: policyVersion, tool_category: category })

    // Local read-only: always allow. The model is still asked (the label is logged, for the report), but with a shorter timeout, and a timeout is also allowed
    // v2.1: if the model explicitly judges it UNAUTHORIZED, do not auto-allow; treat it as a "local write" and follow the v2 rules below (HIGH: deny; LOW: ask the human / deny when out of budget)
    //       Model timeout/error, or judged AUTHORIZED or UNCERTAIN: still allow
    let readCls = null
    if (category === 'local_read') {
      const cls = await callClassifier(info, meta, cfg.policy?.local_read_classifier_timeout_ms ?? 10000)
      recordClassifier(record, cls)
      if (!(policyVersion === 'v2.1' && cls.ok && cls.authorization === 'UNAUTHORIZED')) {
        Object.assign(record, { policy_decision: 'allow', policy_reason: 'local_read', escalated: false })
        return 'allow'
      }
      readCls = cls
      record.flagged_read = true
    }

    // The same call was decided before: reuse it directly; do not ask the model, do not ask the human, do not use budget
    const key = `${info.sessionId}|${info.toolName}|${stableStringify(info.toolArgs)}`
    if (decided.has(key)) {
      Object.assign(record, { policy_decision: decided.get(key), policy_reason: 'repeat_call', escalated: false })
      return decided.get(key)
    }

    const cls = readCls ?? await callClassifier(info, meta)
    if (!readCls) recordClassifier(record, cls)
    const policyMeta = readCls ? { ...meta, read_only: false, touches_remote: false } : meta
    const { decision, reason } = decideV2({ cls, meta: policyMeta, budgetLeft: budget.left(info.sessionId) }, cfg)
    Object.assign(record, { policy_decision: decision, policy_reason: reason })

    let final = decision === 'allow' ? 'allow' : 'deny'
    if (decision === 'ask') {
      budget.consume(info.sessionId)
      // The reviewer does not see the safety gate model's label / rationale; it judges independently
      const r = await review(info, cfg)
      Object.assign(record, {
        escalated: true, reviewer_approved: r.approved, reviewer_reason: r.reason, reviewer_usage: r.usage,
        reviewer_error: r.error, reviewer_latency_ms: r.latencyMs,
      })
      final = r.approved ? 'allow' : 'deny'
    } else {
      record.escalated = false
      if (reason === 'no_budget' || reason === 'reserved_for_external') record.budget_exhausted = true
    }
    decided.set(key, final)
    return final
  }

  ctx.on('tools/pre-execute', async (payload, next) => {
    // Only AgentDojo tools are approved (names start with gated_tool_prefix).
    // dsh built-in tools (read, glob, grep, pwsh, etc.) can read any file on the computer,
    // and the agent uses them to peek at AgentDojo task answers and this plugin's code/logs, so they are disabled by default (other_tools = "deny"):
    // no model call, no budget used, logged as non_agentdojo_tool. Set it to "allow" to allow them directly.
    const prefix = cfg.gated_tool_prefix
    if (prefix && !(typeof payload?.name === 'string' && payload.name.startsWith(prefix))) {
      if (cfg.other_tools === 'allow') return await next()
      log({
        event: 'non_agentdojo_tool',
        session_id: payload?.agent?.session?.header?.id ?? payload?.agent?.id,
        call_id: payload?.callId,
        tool_name: payload?.name,
        final_decision: 'deny',
      })
      return { kind: 'deny', reason: NOT_AVAILABLE_MESSAGE }
    }

    const started = Date.now()
    const record = { event: 'decision' }
    let final = 'deny'

    try {
      // 1. Get the data
      const info = extract(payload, cfg)
      const { meta, found } = lookupMetadata(info.toolName)
      Object.assign(record, {
        session_id: info.sessionId,
        call_id: info.callId,
        tool_name: info.toolName,
        raw_tool_name: info.rawToolName,
        tool_args: info.toolArgs,
        tool_metadata: meta,
        metadata_found: found,
        task: info.task,
        trace_len: info.trace.length,
        k: cfg.k,
        budget_before: budget.left(info.sessionId),
      })

      // Two baselines (the control groups required by the assignment): do not call the safety gate model
      //   approve_all : allow every AgentDojo call (equivalent to no safety gate, but dsh built-in tools are still disabled, so the comparison is fair)
      //   escalate_all: send every AgentDojo call to the reviewer; deny once the budget is used up
      if (policyVersion === 'approve_all' || policyVersion === 'escalate_all') {
        Object.assign(record, { policy_version: policyVersion, tool_category: toolCategory(meta) })
        if (policyVersion === 'approve_all') {
          Object.assign(record, { policy_decision: 'allow', policy_reason: 'approve_all', escalated: false })
          final = 'allow'
        } else if (budget.consume(info.sessionId)) {
          const r = await review(info, cfg)
          Object.assign(record, {
            policy_decision: 'ask', policy_reason: 'escalate_all', escalated: true,
            reviewer_approved: r.approved, reviewer_reason: r.reason, reviewer_usage: r.usage,
            reviewer_error: r.error, reviewer_latency_ms: r.latencyMs,
          })
          final = r.approved ? 'allow' : 'deny'
        } else {
          Object.assign(record, { policy_decision: 'deny', policy_reason: 'no_budget', escalated: false, budget_exhausted: true })
          final = 'deny'
        }
        record.budget_after = budget.left(info.sessionId)
        record.final_decision = final
        record.gate_latency_ms = Date.now() - started
        log(record)
        if (final === 'allow') return await next()
        return { kind: 'deny', reason: DENY_MESSAGE }
      }

      if (policyVersion === 'v2' || policyVersion === 'v2.1') {
        final = await decideWithV2(info, meta, record)
        record.budget_after = budget.left(info.sessionId)
        record.final_decision = final
        record.gate_latency_ms = Date.now() - started
        log(record)
        if (final === 'allow') return await next()
        return { kind: 'deny', reason: DENY_MESSAGE }
      }

      // ---------------- v1 below ----------------
      // 2. Call the safety gate model (request structure aligned with the train.py input)
      const cls = await classify({
        userTask: info.task,
        history: info.history,
        proposedToolCall: { function: info.toolName, args: info.toolArgs },
        toolMetadata: meta,
      }, cfg)
      Object.assign(record, {
        authorization: cls.authorization,
        risk_level: cls.riskLevel,
        risk_factors: cls.riskFactors,
        rationale: cls.rationale,
        classifier_ok: cls.ok,
        classifier_error: cls.error,
        classifier_usage: cls.usage,
        classifier_latency_ms: cls.latencyMs,
      })

      // 3. Pick one of three (on model error, handle according to on_error)
      const policyDecision = cls.ok ? decide(cls, cfg) : cfg.on_error
      record.policy_decision = policyDecision

      // 4. Handle ask
      if (policyDecision === 'ask') {
        if (budget.consume(info.sessionId)) {
          // The reviewer does not see the safety gate model's label / rationale; it judges independently
          const r = await review(info, cfg)
          Object.assign(record, {
            escalated: true, reviewer_approved: r.approved, reviewer_reason: r.reason, reviewer_usage: r.usage,
            reviewer_error: r.error, reviewer_latency_ms: r.latencyMs,
          })
          final = r.approved ? 'allow' : 'deny'
        } else {
          Object.assign(record, { escalated: false, budget_exhausted: true })
          final = 'deny'
        }
      } else {
        record.escalated = false
        final = policyDecision === 'allow' ? 'allow' : 'deny'
      }
      record.budget_after = budget.left(info.sessionId)
    } catch (err) {
      // The plugin itself errored: handle according to on_error (ask is also treated as deny here, to avoid using budget)
      record.plugin_error = String(err?.stack ?? err)
      final = cfg.on_error === 'allow' ? 'allow' : 'deny'
    }

    record.final_decision = final
    record.gate_latency_ms = Date.now() - started
    log(record)

    if (final === 'allow') return await next()
    return { kind: 'deny', reason: DENY_MESSAGE }
  })
}
