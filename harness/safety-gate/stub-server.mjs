// Fake safety gate model service
// Used to test the plugin. Returns the two labels by simple rules:
//   read-only tools        -> AUTHORIZED   / LOW   (policy: allow)
//   affects the outside    -> UNAUTHORIZED / HIGH  (policy: deny)
//   other write actions    -> UNCERTAIN    / LOW   (policy: ask)
// Run: node stub-server.mjs        (default port 8001, change it with the PORT environment variable)
import http from 'node:http'

const PORT = Number(process.env.PORT || 8001)

function reply(res, status, obj) {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(obj))
}

http.createServer((req, res) => {
  if (req.method !== 'POST' || !req.url.startsWith('/v1/chat/completions')) {
    return reply(res, 404, { error: { message: 'not found', type: 'invalid_request_error', code: 404 } })
  }
  let raw = ''
  req.on('data', c => (raw += c))
  req.on('end', () => {
    let body, input
    try {
      body = JSON.parse(raw)
      input = JSON.parse(body.messages.find(m => m.role === 'user').content)
    } catch (err) {
      return reply(res, 400, { error: { message: `bad request: ${err.message}`, type: 'invalid_request_error', code: 400 } })
    }
    const meta = input.tool_metadata || {}
    const toolName = input.proposed_tool_call?.function ?? input.tool_name ?? 'unknown'
    let authorization, risk_level, risk_factors = [], rationale = ''
    if (meta.read_only) { authorization = 'AUTHORIZED'; risk_level = 'LOW'; rationale = 'Read-only tool.' }
    else if (meta.touches_remote) { authorization = 'UNAUTHORIZED'; risk_level = 'HIGH'; risk_factors = ['external_side_effect']; rationale = 'Tool has remote side effects.' }
    else { authorization = 'UNCERTAIN'; risk_level = 'LOW'; rationale = 'Write tool; needs review.' }

    console.log(`[stub] ${toolName} -> ${authorization} / ${risk_level}  (history ${input.history?.length ?? 0} entries)`)
    reply(res, 200, {
      id: 'chatcmpl-stub-' + Date.now(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: body.model,
      choices: [{ index: 0, message: { role: 'assistant', content: JSON.stringify({ authorization, risk_level, risk_factors, rationale }) }, finish_reason: 'stop' }],
      usage: { prompt_tokens: Math.ceil(raw.length / 4), completion_tokens: 14, total_tokens: Math.ceil(raw.length / 4) + 14 },
    })
  })
}).listen(PORT, '127.0.0.1', () => console.log(`[stub] fake model service started: http://127.0.0.1:${PORT}/v1/chat/completions`))
