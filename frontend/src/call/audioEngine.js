// Sequential Web Audio playback for streamed TTS chunks.
//
// Why Web Audio (not <audio>/setTimeout): lip-sync must be sampled against the
// real playback clock. We schedule each chunk on an AudioContext and expose the
// currently-active viseme via `currentViseme()`, computed from
// AudioContext.currentTime - never setTimeout.

import { buildVisemeTimeline, makeVisemeCursor } from './lipsync.js'

export class AudioEngine {
  constructor() {
    this.ctx = null
    this.analyser = null
    this.queue = [] // [{ buffer, timeline }]
    this.current = null // { source, startAt, cursor }
    this.playing = false
    this.onStart = null // () => void  (fires when first chunk of a reply starts)
    this.onDrain = null // () => void  (fires when queue empties)
    this._started = false
  }

  async ensure() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)()
      this.analyser = this.ctx.createAnalyser()
      this.analyser.fftSize = 256
      this.analyser.connect(this.ctx.destination)
    }
    if (this.ctx.state === 'suspended') await this.ctx.resume()
  }

  // Decode a WAV (ArrayBuffer) and enqueue it with its viseme timeline.
  async enqueue(arrayBuffer, timings) {
    await this.ensure()
    const buffer = await this.ctx.decodeAudioData(arrayBuffer.slice(0))
    this.queue.push({ buffer, timeline: buildVisemeTimeline(timings) })
    if (!this.playing) this._playNext()
  }

  _playNext() {
    const item = this.queue.shift()
    if (!item) {
      this.playing = false
      this._started = false
      this.current = null
      this.onDrain && this.onDrain()
      return
    }
    if (!this.playing && !this._started) {
      this._started = true
      this.onStart && this.onStart()
    }
    this.playing = true
    const source = this.ctx.createBufferSource()
    source.buffer = item.buffer
    source.connect(this.analyser)
    const startAt = this.ctx.currentTime + 0.02
    source.start(startAt)
    source.onended = () => {
      if (this.current && this.current.source === source) this._playNext()
    }
    this.current = { source, startAt, cursor: makeVisemeCursor(item.timeline) }
  }

  // The viseme to render right now (based on the real audio clock).
  currentViseme() {
    if (!this.current || !this.ctx) return 'closed'
    const elapsed = this.ctx.currentTime - this.current.startAt
    if (elapsed < 0) return 'closed'
    return this.current.cursor(elapsed)
  }

  // 0..1 audio amplitude for the photo-mode reactive glow.
  amplitude() {
    if (!this.analyser) return 0
    const buf = new Uint8Array(this.analyser.frequencyBinCount)
    this.analyser.getByteTimeDomainData(buf)
    let sum = 0
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128
      sum += v * v
    }
    return Math.min(1, Math.sqrt(sum / buf.length) * 3)
  }

  get isPlaying() {
    return this.playing
  }

  // Barge-in: stop immediately and drop everything queued.
  flush() {
    if (this.current && this.current.source) {
      try {
        this.current.source.onended = null
        this.current.source.stop()
      } catch {
        /* ignore */
      }
    }
    this.queue = []
    this.current = null
    this.playing = false
    this._started = false
  }

  close() {
    this.flush()
    if (this.ctx) {
      this.ctx.close().catch(() => {})
      this.ctx = null
    }
  }
}
