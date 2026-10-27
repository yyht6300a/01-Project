// One budget counter per task (dsh session): k when first seen, minus 1 each time a human is asked.
// k set to null means unlimited.
export class Budget {
  constructor(k) {
    this.k = k
    this.remaining = new Map()
  }
  left(sessionId) {
    if (this.k === null || this.k === undefined) return Infinity
    if (!this.remaining.has(sessionId)) this.remaining.set(sessionId, this.k)
    return this.remaining.get(sessionId)
  }
  consume(sessionId) {
    const left = this.left(sessionId)
    if (left <= 0) return false
    if (left !== Infinity) this.remaining.set(sessionId, left - 1)
    return true
  }
}
