import { useEffect, useRef, useState } from 'react'
import { VISEMES } from '../call/lipsync.js'

// Canvas renderer for the layered 2D sprite avatar. This is the only place the
// avatar animates. Falls back to "photo mode" (static profile pic + audio-
// reactive glow) if a persona has no sprite layers.
//
// Props:
//   persona        - has avatar_id + photo_url
//   emotion        - current emotion string
//   speaking       - boolean: is she talking right now
//   getViseme()    - returns the current viseme (sampled from the audio clock)
//   getAmplitude() - 0..1 audio level (for photo-mode glow)

const REQUIRED = ['base', 'eyes_open', 'eyes_closed', ...VISEMES.map((v) => `mouth_${v}`)]

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => resolve(null)
    img.src = src
  })
}

async function urlExists(url) {
  try {
    const res = await fetch(url, { method: 'HEAD' })
    const type = res.headers.get('content-type') || ''
    // The SPA catch-all returns index.html (200) for missing files — require a
    // video content-type so we don't mistake the fallback page for a clip.
    return res.ok && !type.includes('text/html')
  } catch {
    return false
  }
}

export default function AnimatedAvatar({ persona, emotion, speaking, getViseme, getAmplitude }) {
  const canvasRef = useRef(null)
  const [layers, setLayers] = useState(null)
  const [mode, setMode] = useState('loading') // loading | video | sprite | photo
  const photoMode = mode === 'photo'
  const stateRef = useRef({ emotion, speaking })
  stateRef.current = { emotion, speaking }

  // --- pick render mode: video loops > sprite layers > static photo ---
  useEffect(() => {
    let alive = true
    const id = persona?.avatar_id
    if (!id) {
      setMode('photo')
      return
    }
    const base = `/avatars/${id}`
    ;(async () => {
      // Best fidelity first: low-res video loops (idle.mp4 + talking.mp4).
      const [idleOk, talkOk] = await Promise.all([
        urlExists(`${base}/idle.mp4`),
        urlExists(`${base}/talking.mp4`),
      ])
      if (!alive) return
      if (idleOk && talkOk) {
        setMode('video')
        return
      }
      const entries = await Promise.all(
        REQUIRED.map(async (name) => [name, await loadImage(`${base}/${name}.png`)])
      )
      if (!alive) return
      const map = Object.fromEntries(entries)
      if (!map.base) {
        setMode('photo') // no art -> photo fallback
        return
      }
      // optional per-emotion eyes
      for (const emo of ['happy', 'laughing', 'shy', 'sad', 'surprised', 'love']) {
        map[`eyes_${emo}`] = await loadImage(`${base}/eyes_${emo}.png`)
      }
      setLayers(map)
      setMode('sprite')
    })()
    return () => {
      alive = false
    }
  }, [persona?.avatar_id])

  // --- photo mode image ---
  const photoRef = useRef(null)
  useEffect(() => {
    if (!photoMode || !persona?.photo_url) return
    loadImage(`${persona.photo_url}?v=${persona._v || 0}`).then((img) => {
      photoRef.current = img
    })
  }, [photoMode, persona?.photo_url, persona?._v])

  // --- animation loop (sprite/photo modes only) ---
  useEffect(() => {
    if (mode === 'video' || mode === 'loading') return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let raf
    const blink = { next: performance.now() + rand(2000, 6000), until: 0 }
    const particles = []

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const size = Math.min(canvas.clientWidth, canvas.clientHeight)
      canvas.width = size * dpr
      canvas.height = size * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const draw = (now) => {
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      ctx.clearRect(0, 0, w, h)
      const { emotion: emo, speaking: spk } = stateRef.current
      const t = now / 1000

      // idle breathing + emotion head motion
      const breathe = Math.sin(t * 1.6) * 0.006
      let bobY = Math.sin(t * 1.6) * h * 0.008
      let tilt = 0
      let scale = 1 + breathe
      if (emo === 'laughing') { bobY += Math.sin(t * 9) * h * 0.012; tilt = Math.sin(t * 9) * 0.03 }
      else if (emo === 'happy') bobY += Math.sin(t * 4) * h * 0.006
      else if (emo === 'love') tilt = Math.sin(t * 2) * 0.04
      else if (emo === 'shy') { tilt = 0.06; bobY += h * 0.01 }
      else if (emo === 'sad') { tilt = -0.05; bobY += h * 0.015 }
      else if (emo === 'surprised') scale += 0.03

      if (photoMode) {
        drawPhoto(ctx, w, h, photoRef.current, getAmplitude?.() || 0, spk, t)
      } else if (layers) {
        ctx.save()
        ctx.translate(w / 2, h / 2 + bobY)
        ctx.rotate(tilt)
        ctx.scale(scale, scale)
        ctx.translate(-w / 2, -h / 2)
        const rect = [0, 0, w, h]
        ctx.drawImage(layers.base, ...rect)

        // eyes: blink overrides; else emotion variant or open
        let blinking = now < blink.until
        if (now > blink.next) {
          blink.until = now + 120
          blink.next = now + rand(2000, 6000)
        }
        const eyeVariant = layers[`eyes_${emo}`]
        const eyeImg = blinking ? layers.eyes_closed : eyeVariant || layers.eyes_open
        if (eyeImg) ctx.drawImage(eyeImg, ...rect)

        // mouth: viseme while speaking, else closed
        const viseme = spk ? getViseme?.() || 'closed' : 'closed'
        const mouthImg = layers[`mouth_${viseme}`] || layers.mouth_closed
        if (mouthImg) ctx.drawImage(mouthImg, ...rect)
        ctx.restore()
      }

      // particles (hearts / sparkles) during expressive speech
      if (spk && (emo === 'love' || emo === 'happy' || emo === 'laughing')) {
        if (Math.random() < 0.18) particles.push(spawnParticle(w, h, emo))
      }
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]
        p.y -= p.vy
        p.x += Math.sin(p.y * 0.05) * 0.4
        p.life -= 0.012
        if (p.life <= 0) { particles.splice(i, 1); continue }
        ctx.globalAlpha = Math.max(0, p.life)
        ctx.font = `${p.size}px serif`
        ctx.fillText(p.glyph, p.x, p.y)
        ctx.globalAlpha = 1
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [layers, mode, photoMode, getViseme, getAmplitude])

  if (mode === 'video') {
    return <VideoAvatar avatarId={persona.avatar_id} speaking={speaking} />
  }
  return <canvas ref={canvasRef} className="avatar-canvas" />
}

// Two low-res video loops cross-faded in real time: idle.mp4 plays when quiet,
// talking.mp4 while she speaks. Cheap (plain <video> decode), works on modest
// hardware, and looks far more lifelike than the sprite. Clips are generated
// offline once — see frontend/public/avatars/README.md.
function VideoAvatar({ avatarId, speaking }) {
  const idleRef = useRef(null)
  const talkRef = useRef(null)

  useEffect(() => {
    // Keep both playing so the swap is an instant opacity cross-fade.
    idleRef.current?.play().catch(() => {})
    talkRef.current?.play().catch(() => {})
  }, [])

  return (
    <div className="avatar-video-wrap">
      <video
        ref={idleRef}
        src={`/avatars/${avatarId}/idle.mp4`}
        muted
        loop
        playsInline
        autoPlay
        style={{ opacity: speaking ? 0 : 1 }}
      />
      <video
        ref={talkRef}
        src={`/avatars/${avatarId}/talking.mp4`}
        muted
        loop
        playsInline
        autoPlay
        style={{ opacity: speaking ? 1 : 0 }}
      />
    </div>
  )
}

function drawPhoto(ctx, w, h, img, amp, speaking, t) {
  const cx = w / 2
  const cy = h / 2
  const r = Math.min(w, h) * 0.34
  const glow = speaking ? 0.25 + amp * 0.75 : 0.12
  // audio-reactive glow ring
  ctx.save()
  ctx.shadowColor = `rgba(0,168,132,${glow})`
  ctx.shadowBlur = 30 + glow * 60
  ctx.beginPath()
  ctx.arc(cx, cy + Math.sin(t * 1.4) * h * 0.006, r + glow * 8, 0, Math.PI * 2)
  ctx.fillStyle = 'rgba(0,0,0,0.001)'
  ctx.fill()
  ctx.restore()
  if (!img) return
  ctx.save()
  ctx.beginPath()
  ctx.arc(cx, cy + Math.sin(t * 1.4) * h * 0.006, r, 0, Math.PI * 2)
  ctx.clip()
  const scale = 1 + (speaking ? amp * 0.03 : 0) + Math.sin(t * 1.4) * 0.004
  const dw = r * 2 * scale
  ctx.drawImage(img, cx - dw / 2, cy - dw / 2 + Math.sin(t * 1.4) * h * 0.006, dw, dw)
  ctx.restore()
}

function spawnParticle(w, h, emo) {
  const hearts = ['❤️', '💕', '💗']
  const sparkles = ['✨', '⭐', '💫']
  const set = emo === 'love' ? hearts : sparkles
  return {
    x: w * (0.3 + Math.random() * 0.4),
    y: h * 0.6,
    vy: 0.6 + Math.random() * 0.8,
    size: 20 + Math.random() * 18,
    glyph: set[(Math.random() * set.length) | 0],
    life: 1,
  }
}

function rand(a, b) {
  return a + Math.random() * (b - a)
}
