// Maps Phase 4 `timings` metadata to the 6 mouth visemes and builds a flat
// timeline the avatar samples by AudioContext time during playback.
//
// Visemes: closed, slightly_open, open, wide, rounded, teeth

export const VISEMES = ['closed', 'slightly_open', 'open', 'wide', 'rounded', 'teeth']

// Character -> viseme. A cheap but effective approximation of mouth shapes.
export function visemeForChar(ch) {
  const c = ch.toLowerCase()
  if (c === 'a') return 'wide'
  if (c === 'o' || c === 'u' || c === 'w') return 'rounded'
  if (c === 'e' || c === 'i') return 'teeth'
  if ('mbp'.includes(c)) return 'closed'
  if ('fv'.includes(c)) return 'teeth'
  if ('ldtnszcgkjrhy'.includes(c)) return 'slightly_open'
  if ('qx'.includes(c)) return 'open'
  if (/[a-z]/.test(c)) return 'open'
  return 'closed' // spaces, punctuation, digits
}

// Build a [{ t, d, viseme }] timeline from a timings object. Word-level units
// are expanded into per-letter sub-units so the mouth still moves within a word.
export function buildVisemeTimeline(timings) {
  if (!timings || !Array.isArray(timings.units)) return []
  const out = []
  for (const u of timings.units) {
    const s = u.s || ''
    const letters = [...s]
    if (letters.length <= 1) {
      out.push({ t: u.t, d: u.d, viseme: visemeForChar(s) })
      continue
    }
    // Split this unit's duration across its letters (skip pure spaces).
    const per = u.d / letters.length
    letters.forEach((ch, i) => {
      out.push({ t: u.t + i * per, d: per, viseme: visemeForChar(ch) })
    })
  }
  return out
}

// Pick the viseme active at time `elapsed` (seconds) within a chunk's timeline.
// Uses a small cursor for O(1) amortized lookup during the rAF loop.
export function makeVisemeCursor(timeline) {
  let i = 0
  return (elapsed) => {
    if (!timeline.length) return 'closed'
    // advance
    while (i < timeline.length && timeline[i].t + timeline[i].d < elapsed) i++
    // rewind if playback jumped back (new chunk)
    while (i > 0 && timeline[i - 1].t > elapsed) i--
    const u = timeline[Math.min(i, timeline.length - 1)]
    if (elapsed < u.t) return 'closed'
    return u.viseme
  }
}
