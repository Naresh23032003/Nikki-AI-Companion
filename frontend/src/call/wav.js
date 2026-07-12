// Encode a Float32 PCM buffer (as produced by the VAD) into a 16-bit WAV, then
// base64 — the server's faster-whisper decodes WAV fine via PyAV.

export function float32ToWav(float32, sampleRate = 16000) {
  const len = float32.length
  const buffer = new ArrayBuffer(44 + len * 2)
  const view = new DataView(buffer)

  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i))
  }

  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + len * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeStr(36, 'data')
  view.setUint32(40, len * 2, true)

  let off = 44
  for (let i = 0; i < len; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]))
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    off += 2
  }
  return new Uint8Array(buffer)
}

export function bytesToBase64(bytes) {
  let bin = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk))
  }
  return btoa(bin)
}

// Concatenate multiple Float32 arrays (used to merge paused-mid-thought fragments).
export function concatFloat32(arrays) {
  const total = arrays.reduce((n, a) => n + a.length, 0)
  const out = new Float32Array(total)
  let off = 0
  for (const a of arrays) {
    out.set(a, off)
    off += a.length
  }
  return out
}
