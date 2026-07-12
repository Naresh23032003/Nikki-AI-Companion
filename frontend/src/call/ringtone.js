// A soft, synthesized ringtone (no audio asset). Two gentle tones that pulse
// while the call connects/rings. Started from a user gesture so autoplay is OK.

export function makeRingtone() {
  let ctx = null
  let timer = null
  let stopped = true

  const beep = () => {
    if (!ctx || stopped) return
    const now = ctx.currentTime
    const gain = ctx.createGain()
    gain.connect(ctx.destination)
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(0.12, now + 0.05)
    gain.gain.linearRampToValueAtTime(0.12, now + 0.4)
    gain.gain.linearRampToValueAtTime(0, now + 0.55)
    for (const freq of [440, 554]) {
      const osc = ctx.createOscillator()
      osc.type = 'sine'
      osc.frequency.value = freq
      osc.connect(gain)
      osc.start(now)
      osc.stop(now + 0.55)
    }
  }

  return {
    start() {
      if (!stopped) return
      stopped = false
      ctx = new (window.AudioContext || window.webkitAudioContext)()
      if (ctx.state === 'suspended') ctx.resume().catch(() => {})
      beep()
      timer = setInterval(beep, 2000) // ring cadence
    },
    stop() {
      stopped = true
      clearInterval(timer)
      if (ctx) {
        ctx.close().catch(() => {})
        ctx = null
      }
    },
  }
}
