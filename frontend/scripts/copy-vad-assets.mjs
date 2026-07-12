// Copies the VAD worklet, Silero ONNX model, and onnxruntime-web WASM binaries
// into public/vad/ so they're served same-origin. This keeps Call-mode VAD
// working offline / on the LAN / inside the PWA (no CDN fetches, CSP-friendly).
//
// Runs automatically via the "prebuild"/"predev" npm scripts.
import { cpSync, mkdirSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')
const dest = join(root, 'public', 'vad')
mkdirSync(dest, { recursive: true })

const vadDist = join(root, 'node_modules', '@ricky0123', 'vad-web', 'dist')
const ortDist = join(root, 'node_modules', 'onnxruntime-web', 'dist')

const copy = (from, to) => {
  cpSync(from, join(dest, to), { force: true })
  console.log('  vad-assets:', to)
}

// VAD worklet + models
copy(join(vadDist, 'vad.worklet.bundle.min.js'), 'vad.worklet.bundle.min.js')
copy(join(vadDist, 'silero_vad_v5.onnx'), 'silero_vad_v5.onnx')
copy(join(vadDist, 'silero_vad_legacy.onnx'), 'silero_vad_legacy.onnx')

// onnxruntime-web wasm/mjs binaries (loaded at runtime by ort)
for (const f of readdirSync(ortDist)) {
  if (f.startsWith('ort-wasm') && (f.endsWith('.wasm') || f.endsWith('.mjs'))) {
    copy(join(ortDist, f), f)
  }
}

console.log('VAD assets ready in public/vad/')
