import { useEffect, useRef, useState } from 'react'
import Avatar from './Avatar.jsx'
import { api } from '../api.js'

export default function SettingsPanel({ persona, onClose, onPersonaChange, onClearChat, sourceFilter, onSourceFilterChange, onOpenJournal }) {
  const [personas, setPersonas] = useState([])
  const [activeId, setActiveId] = useState(persona?.id)
  const [memories, setMemories] = useState([])
  const [busy, setBusy] = useState(false)
  const [proactive, setProactive] = useState(null) // { enabled, paused_until, ... }
  const [relationship, setRelationship] = useState(null)
  const [brain, setBrain] = useState(null)
  const [dayState, setDayState] = useState(null)
  const [voice, setVoice] = useState(null)
  const [bench, setBench] = useState(null)
  const [benchBusy, setBenchBusy] = useState(false)
  const [devStage, setDevStage] = useState('')
  const [devAffection, setDevAffection] = useState('')
  const fileRef = useRef(null)

  useEffect(() => {
    api.getPersonas().then((d) => {
      setPersonas(d.personas || [])
      setActiveId(d.active)
    }).catch(() => {})
    refreshMemories()
    api.proactiveStatus().then(setProactive).catch(() => {})
    api.getRelationship().then(setRelationship).catch(() => {})
    api.brainStatus().then(setBrain).catch(() => {})
    api.dayState().then(setDayState).catch(() => {})
    api.voiceStatus().then(setVoice).catch(() => {})
  }, [])

  const runBench = async () => {
    setBenchBusy(true)
    try {
      setBench(await api.voiceBench(['neutral', 'happy', 'sad']))
    } catch {
      /* engines not ready */
    } finally {
      setBenchBusy(false)
    }
  }

  const applyDevOverride = async () => {
    const payload = {}
    if (devStage) payload.stage = devStage
    if (devAffection !== '') payload.affection = Number(devAffection)
    if (!Object.keys(payload).length) return
    try {
      const s = await api.overrideRelationship(payload)
      setRelationship(s)
      setDevStage('')
      setDevAffection('')
    } catch {
      /* invalid input */
    }
  }

  const pauseProactive = async (hours) => {
    try {
      await api.proactivePause(hours)
      const s = await api.proactiveStatus()
      setProactive(s)
    } catch {
      /* scheduler not running */
    }
  }

  const refreshMemories = () =>
    api.getMemories().then((d) => setMemories(d.memories || [])).catch(() => {})

  const pickPersona = async (id) => {
    if (id === activeId || busy) return
    setBusy(true)
    try {
      const p = await api.setActivePersona(id)
      setActiveId(id)
      onPersonaChange({ ...p, _v: Date.now() })
    } finally {
      setBusy(false)
    }
  }

  const onPhoto = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true)
    try {
      const p = await api.uploadPhoto(file)
      onPersonaChange({ ...p, _v: Date.now() })
    } finally {
      setBusy(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const removeMemory = async (id) => {
    await api.deleteMemory(id).catch(() => {})
    setMemories((m) => m.filter((x) => x.id !== id))
  }

  const completeMemory = async (id) => {
    try {
      const updated = await api.completeMemory(id)
      setMemories((m) => m.map((x) => (x.id === id ? updated : x)))
    } catch {
      /* ignore */
    }
  }

  const clearChat = async () => {
    if (!confirm('Clear this chat? Messages will be deleted.')) return
    await onClearChat()
    onClose()
  }

  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <div className="sheet-head">
          <button className="icon-btn" onClick={onClose} aria-label="Back">
            <BackIcon />
          </button>
          <h2>Settings</h2>
        </div>

        <div className="sheet-body">
          {/* Profile photo */}
          <section className="settings-section profile-row">
            <Avatar persona={persona} size={64} />
            <div className="profile-meta">
              <div className="profile-name">{persona?.name}</div>
              <button
                className="link-btn"
                disabled={busy}
                onClick={() => fileRef.current?.click()}
              >
                Change profile photo
              </button>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                hidden
                onChange={onPhoto}
              />
              <div className="hint">Use the same photo as her WhatsApp account.</div>
            </div>
          </section>

          {/* Persona picker */}
          <section className="settings-section">
            <h3>Persona</h3>
            <div className="persona-list">
              {personas.map((p) => (
                <button
                  key={p.id}
                  className={`persona-item ${p.id === activeId ? 'active' : ''}`}
                  onClick={() => pickPersona(p.id)}
                  disabled={busy}
                >
                  <Avatar persona={p} size={40} />
                  <span>{p.name}</span>
                  {p.id === activeId && <CheckIcon />}
                </button>
              ))}
            </div>
          </section>

          {/* Relationship */}
          {relationship && (
            <section className="settings-section">
              <h3>Relationship</h3>
              <div className="rel-row">
                <span className={`stage-badge ${relationship.stage}`}>
                  {relationship.stage}
                </span>
                <span className="hint">
                  {relationship.days_known} day{relationship.days_known === 1 ? '' : 's'} ·{' '}
                  {relationship.meaningful_exchanges} meaningful exchanges ·{' '}
                  {relationship.memory_count} memories
                </span>
              </div>
              <div className="affection-bar" title={`Affection ${Math.round(relationship.affection)}/100`}>
                <div
                  className="affection-fill"
                  style={{ width: `${Math.min(100, relationship.affection)}%` }}
                />
              </div>
              <div className="hint">Affection {Math.round(relationship.affection)}/100 - grows slowly with real conversations.</div>

              <details className="dev-tools">
                <summary className="hint">dev override (testing only)</summary>
                <div className="dev-row">
                  <select value={devStage} onChange={(e) => setDevStage(e.target.value)}>
                    <option value="">stage…</option>
                    {['stranger', 'acquaintance', 'friend', 'close', 'girlfriend'].map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    placeholder="affection"
                    value={devAffection}
                    onChange={(e) => setDevAffection(e.target.value)}
                  />
                  <button className="pill-btn" onClick={applyDevOverride}>Apply</button>
                </div>
              </details>
            </section>
          )}

          {/* Proactive messaging */}
          {proactive?.enabled && (
            <section className="settings-section">
              <h3>Her check-in texts</h3>
              {proactive.paused_until ? (
                <div className="hint" style={{ marginBottom: 8 }}>
                  Paused until {new Date(proactive.paused_until).toLocaleString()}
                </div>
              ) : (
                <div className="hint" style={{ marginBottom: 8 }}>
                  She may text you first during {proactive.active_hours}
                </div>
              )}
              <div className="pause-row">
                {proactive.paused_until ? (
                  <button className="link-btn" onClick={() => pauseProactive(0)}>
                    Resume
                  </button>
                ) : (
                  <>
                    <button className="pill-btn" onClick={() => pauseProactive(4)}>
                      Pause 4h
                    </button>
                    <button className="pill-btn" onClick={() => pauseProactive(12)}>
                      Pause 12h
                    </button>
                    <button className="pill-btn" onClick={() => pauseProactive(24)}>
                      Pause 24h
                    </button>
                  </>
                )}
              </div>
            </section>
          )}

          {/* Voice system */}
          {voice && (
            <section className="settings-section">
              <h3>Voice</h3>
              {voice.training && !voice.training.done && (
                <div className="hint" style={{ marginBottom: 6 }}>
                  🏋️ training: {voice.training.last_stage || 'starting…'}
                  {voice.training.last_epoch_line && (
                    <div>{voice.training.last_epoch_line}</div>
                  )}
                </div>
              )}
              {voice.training?.done && (
                <div className="hint" style={{ marginBottom: 6 }}>✅ RVC training complete</div>
              )}
              <div className="hint">
                calls: <b>{voice.call_voice}</b>
                {' · '}rvc {voice.rvc_ready ? '✓ ready' : 'not trained'}
                {voice.rvc_last_latency_ms != null && ` (${Math.round(voice.rvc_last_latency_ms)}ms)`}
                {' · '}studio {voice.studio_installed ? `✓ ${voice.studio_engine}` : 'not installed'}
              </div>
              {voice.song_library?.length > 0 && (
                <div className="hint">songs: {voice.song_library.join(', ')}</div>
              )}
              <button className="pill-btn" style={{ marginTop: 8 }}
                      disabled={benchBusy} onClick={runBench}>
                {benchBusy ? 'Rendering…' : 'Run voice bench'}
              </button>
              {bench?.renders?.length > 0 && (
                <div className="bench-grid">
                  {bench.renders.map((r, i) => (
                    <div key={i} className="bench-row">
                      <span className="hint">{r.emotion} #{r.line}</span>
                      {r.call_url && <audio controls src={r.call_url} preload="none" />}
                      {r.studio_url && <audio controls src={r.studio_url} preload="none" />}
                      {r.studio_error && <span className="hint">studio: {r.studio_error}</span>}
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Brain: cloud usage, routing, deferred */}
          {brain && (
            <section className="settings-section">
              <h3>Brain</h3>
              <div className="hint">
                Cloud: {brain.cloud_enabled ? 'on' : 'off'}
                {brain.usage?.degraded && ' · degraded (budget) - casual stays local'}
              </div>
              {brain.usage && (
                <>
                  <div className="brain-row">
                    <span>Requests today</span>
                    <span>{brain.usage.requests} / {brain.usage.requests_budget}</span>
                  </div>
                  <div className="brain-row">
                    <span>Tokens today</span>
                    <span>{brain.usage.tokens} / {brain.usage.tokens_budget}</span>
                  </div>
                  <div className="affection-bar">
                    <div className="affection-fill" style={{ width: `${Math.min(100, (brain.usage.fraction || 0) * 100)}%` }} />
                  </div>
                </>
              )}
              <div className="hint" style={{ marginTop: 6 }}>
                Providers: {(brain.providers || []).map((p) =>
                  `${p.name}${p.enabled ? (p.has_key ? ' ✓' : ' (no key)') : ' (off)'}`).join(' · ')}
              </div>
              {brain.routing && Object.keys(brain.routing).length > 0 && (
                <div className="hint" style={{ marginTop: 4 }}>
                  Routing: {Object.entries(brain.routing).map(([k, v]) => `${k} ${v}`).join(' · ')}
                </div>
              )}
              {brain.deferred?.length > 0 && (
                <div className="hint" style={{ marginTop: 4 }}>
                  Deferred: {brain.deferred.filter((d) => !d.done).length} pending
                </div>
              )}
            </section>
          )}

          {/* Her day (dev view) */}
          {dayState && (
            <section className="settings-section">
              <h3>Her day (dev)</h3>
              <div className="hint">
                mood <b>{dayState.mood}</b> · energy {dayState.energy}/5
                {dayState.on_mind && <> · on her mind: {dayState.on_mind}</>}
              </div>
              {dayState.slots && (
                <div className="hint" style={{ marginTop: 4 }}>
                  {['morning', 'afternoon', 'evening'].map((s) => (
                    <div key={s}><b>{s}:</b> {dayState.slots[s]}</div>
                  ))}
                </div>
              )}
              <button
                className="pill-btn"
                style={{ marginTop: 8 }}
                onClick={() => api.regenerateDayState().then(setDayState).catch(() => {})}
              >
                Regenerate day
              </button>
            </section>
          )}

          {/* Mood journal */}
          <section className="settings-section">
            <h3>Mood journal</h3>
            <div className="hint" style={{ marginBottom: 8 }}>
              A private, passive record of how you've been - inferred from your conversations, never asked.
            </div>
            <button className="pill-btn" onClick={onOpenJournal}>Open journal</button>
          </section>

          {/* History view */}
          <section className="settings-section">
            <h3>History view</h3>
            <div className="toggle-row">
              <span>Filter by source</span>
              <select className="source-filter" value={sourceFilter} onChange={(e) => onSourceFilterChange(e.target.value)}>
                <option value="all">All sources</option>
                <option value="webapp_chat">Web app</option>
                <option value="webapp_call">Call</option>
                <option value="whatsapp">WhatsApp</option>
                <option value="tablet">Tablet</option>
                <option value="iot">IoT</option>
              </select>
            </div>
            <div className="hint" style={{ marginTop: 8 }}>
              Source legend: <span className="legend-dot webapp">●</span> web app, <span className="legend-dot call">●</span> call, <span className="legend-dot whatsapp">●</span> WhatsApp, <span className="legend-dot tablet">●</span> tablet, <span className="legend-dot iot">●</span> IoT.
            </div>
          </section>

          {/* Memories */}
          <section className="settings-section">
            <h3>Memories ({memories.length})</h3>
            {memories.length === 0 && <div className="hint">No memories yet.</div>}
            <ul className="memory-list">
              {memories.map((m) => (
                <li key={m.id} className="memory-item">
                  <div className="memory-text">
                    <span className={`chip ${m.category}`}>{m.category}</span>
                    {m.kind && m.kind !== 'permanent' && (
                      <span className={`chip kind-${m.kind}`}>{m.kind}</span>
                    )}
                    {m.fact}
                  </div>
                  {(m.kind === 'event' || m.kind === 'recurring' || m.kind === 'transient') && (
                    <button
                      className="icon-btn"
                      title="Mark done - stops being mentioned"
                      onClick={() => completeMemory(m.id)}
                      aria-label="Mark memory completed"
                    >
                      ✓
                    </button>
                  )}
                  <button
                    className="icon-btn danger"
                    onClick={() => removeMemory(m.id)}
                    aria-label="Delete memory"
                  >
                    <TrashIcon />
                  </button>
                </li>
              ))}
            </ul>
          </section>

          {/* Danger zone */}
          <section className="settings-section">
            <button className="danger-btn" onClick={clearChat}>
              Clear chat
            </button>
          </section>
        </div>
      </div>
    </div>
  )
}

function BackIcon() {
  return (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
      <path d="M20 11H7.8l5.6-5.6L12 4l-8 8 8 8 1.4-1.4L7.8 13H20z" />
    </svg>
  )
}
function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" className="check">
      <path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" />
    </svg>
  )
}
function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M6 7h12l-1 14H7L6 7zm3-3h6l1 2H8l1-2z" />
    </svg>
  )
}
