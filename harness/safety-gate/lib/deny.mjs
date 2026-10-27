// Message returned to the agent when the plugin denies a call.
// extract.mjs also relies on this sentence to recognize "calls denied by this plugin" and exclude them from the trace.
export const DENY_MESSAGE = 'This tool call was blocked by the safety approval policy.'

// Message returned to the agent when a non-AgentDojo tool (dsh built-in read, glob, grep, pwsh, etc.) is disabled.
// Deliberately different from DENY_MESSAGE: this is not the result of a safety approval; the tool is just unavailable in this environment.
export const NOT_AVAILABLE_MESSAGE = 'This tool is not available in this environment. Use the provided task tools instead.'
