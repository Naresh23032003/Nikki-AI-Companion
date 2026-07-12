// Time / date helpers and the human-like typing delay.

export function timeHHMM(ts) {
  const d = ts ? new Date(ts) : new Date()
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// A WhatsApp-style date separator label: "Today", "Yesterday", or a date.
export function dateLabel(ts) {
  const d = new Date(ts)
  const today = new Date()
  const yesterday = new Date()
  yesterday.setDate(today.getDate() - 1)
  const sameDay = (a, b) =>
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  if (sameDay(d, today)) return 'Today'
  if (sameDay(d, yesterday)) return 'Yesterday'
  return d.toLocaleDateString([], { day: 'numeric', month: 'long', year: 'numeric' })
}

export function dayKey(ts) {
  const d = new Date(ts)
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`
}

// Human-like "typing" delay (ms) proportional to reply length, clamped so it
// never feels instant and never drags. ~45ms/char, roughly reading/typing pace.
export function typingDelay(text) {
  const len = (text || '').length
  return Math.min(3500, Math.max(700, 400 + len * 45))
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// Defensively strip any stray {"emotion": "..."} tag from a reply before we
// show it or speak it (the backend also strips, this is belt-and-suspenders).
const EMOTION_TAG = /\{\s*"?emotion"?\s*:\s*"?[a-zA-Z_]+"?\s*\}/gi
export function stripEmotionTag(text) {
  return (text || '').replace(EMOTION_TAG, '').trim()
}
