// 安全审批门插件
// 每次 tool call 执行前：
//   1. 从 dsh 取数据（工具名、参数、用户任务、trace、会话 ID）
//   2. 查表填 tool_metadata
//   3. 调安全门模型拿 authorization + risk_level
//   4. 固定阈值三选一：allow / deny / ask（这里后面我们具体看怎么做，我这里这么写保证链路完整）
//   5. ask：预算 > 0 交给模拟审核员（预算 -1）；预算 = 0 直接 deny
//   6. 记日志，返回结果给 dsh

import { loadConfig } from './lib/config.mjs'
import { lookupMetadata } from './lib/metadata.mjs'
import { extract } from './lib/extract.mjs'
import { classify } from './lib/classifier.mjs'
import { decide } from './lib/policy.mjs'
import { Budget } from './lib/budget.mjs'
import { review } from './lib/reviewer.mjs'
import { makeLogger } from './lib/logger.mjs'
import { DENY_MESSAGE, NOT_AVAILABLE_MESSAGE } from './lib/deny.mjs'

export const name = 'safety-gate'

export function apply(ctx) {
  const cfg = loadConfig()
  const budget = new Budget(cfg.k)
  const log = makeLogger(cfg.log_dir)
  console.log(`[safety-gate] 已加载。k=${cfg.k}，非 AgentDojo 工具=${cfg.other_tools ?? 'deny'}，模型服务=${cfg.classifier.url}`)

  ctx.on('tools/pre-execute', async (payload, next) => {
    // 只审批 AgentDojo 的工具（名字以 gated_tool_prefix 开头）。
    // dsh 自带工具（read、glob、grep、pwsh 等）能读电脑上的任意文件，
    // agent 会用它们偷看 AgentDojo 的任务答案和本插件的代码/日志，所以默认禁用（other_tools = "deny"）：
    // 不调模型、不消耗预算，日志里记为 non_agentdojo_tool。设为 "allow" 则直接放行。
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
      // 1. 取数据
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

      // 2. 调安全门模型（报文结构与 train.py 输入对齐）
      const cls = await classify({
        userTask: info.task,
        history: info.history,
        proposedToolCall: { function: info.toolName, args: info.toolArgs },
        toolMetadata: meta,
        task: info.task,
        toolName: info.toolName,
        toolArgs: info.toolArgs,
        trace: info.trace,
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

      // 3. 三选一（模型出错时按 on_error 处理）
      const policyDecision = cls.ok ? decide(cls, cfg) : cfg.on_error
      record.policy_decision = policyDecision

      // 4. 处理 ask
      if (policyDecision === 'ask') {
        if (budget.consume(info.sessionId)) {
          const r = await review({ ...info, rationale: cls.rationale }, cfg)
          Object.assign(record, { escalated: true, reviewer_approved: r.approved, reviewer_reason: r.reason, reviewer_usage: r.usage })
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
      // 插件自己出错：按 on_error 处理（ask 在这里也当 deny，避免消耗预算）
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
