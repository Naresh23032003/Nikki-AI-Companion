import { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'

const MOOD_COLORS = {
  happy: '#7cd4a3', content: '#7cd4a3', grateful: '#7cd4a3', proud: '#7cd4a3',
  excited: '#e0b978', hopeful: '#e0b978',
  stressed: '#e08fb0', anxious: '#e08fb0', overwhelmed: '#e08fb0', hurt: '#e08fb0',
  sad: '#86a9e0', lonely: '#86a9e0', tired: '#86a9e0', bored: '#86a9e0',
  frustrated: '#c39ce0', angry: '#c39ce0',
}

function moodColor(label) {
  return MOOD_COLORS[(label || '').toLowerCase()] || '#9fd0d6'
}

function groupByDate(entries) {
  const groups = new Map()
  for (const e of entries) {
    if (!groups.has(e.date)) groups.set(e.date, [])
    groups.get(e.date).push(e)
  }
  return [...groups.entries()]
}

export default function MoodJournal({ onClose }) {
  const [entries, setEntries] = useState([])
  const [loading, setLoading] = useState(true)
  const [moodFilter, setMoodFilter] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [editDraft, setEditDraft] = useState({ mood_label: '', intensity: 3, why: '' })

  const load = () => {
    setLoading(true)
    api.getJournal(null, moodFilter || null)
      .then((d) => setEntries(d.entries || []))
      .catch(() => setEntries([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [moodFilter]) // eslint-disable-line react-hooks/exhaustive-deps

  const moodOptions = useMemo(() => {
    const set = new Set(entries.map((e) => e.mood_label))
    return [...set].sort()
  }, [entries])

  const startEdit = (entry) => {
    setEditingId(entry.id)
    setEditDraft({ mood_label: entry.mood_label, intensity: entry.intensity, why: entry.why })
  }

  const saveEdit = async (id) => {
    try {
      const updated = await api.updateJournalEntry(id, editDraft)
      setEntries((es) => es.map((e) => (e.id === id ? updated : e)))
    } finally {
      setEditingId(null)
    }
  }

  const remove = async (id) => {
    if (!confirm('Delete this journal entry?')) return
    await api.deleteJournalEntry(id).catch(() => {})
    setEntries((es) => es.filter((e) => e.id !== id))
  }

  const grouped = groupByDate(entries)

  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <div className="sheet-head">
          <button className="icon-btn" onClick={onClose} aria-label="Back">
            <BackIcon />
          </button>
          <h2>Mood journal</h2>
        </div>

        <div className="sheet-body">
          <section className="settings-section">
            <div className="toggle-row">
              <span>Filter by mood</span>
              <select className="source-filter" value={moodFilter} onChange={(e) => setMoodFilter(e.target.value)}>
                <option value="">All moods</option>
                {moodOptions.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div className="hint" style={{ marginTop: 8 }}>
              She quietly notices patterns in your day from your conversations — never surveys, never asks.
            </div>
          </section>

          {loading && <div className="hint" style={{ padding: '0 16px' }}>Loading…</div>}
          {!loading && grouped.length === 0 && (
            <div className="hint" style={{ padding: '0 16px' }}>No journal entries yet.</div>
          )}

          {grouped.map(([date, dayEntries]) => (
            <section className="settings-section" key={date}>
              <h3>{new Date(date).toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' })}</h3>
              <ul className="memory-list">
                {dayEntries.map((entry) => (
                  <li key={entry.id} className="memory-item">
                    {editingId === entry.id ? (
                      <div className="dev-row" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 6 }}>
                        <input
                          value={editDraft.mood_label}
                          onChange={(e) => setEditDraft((d) => ({ ...d, mood_label: e.target.value }))}
                          placeholder="mood"
                        />
                        <select
                          value={editDraft.intensity}
                          onChange={(e) => setEditDraft((d) => ({ ...d, intensity: Number(e.target.value) }))}
                        >
                          {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
                        </select>
                        <textarea
                          value={editDraft.why}
                          onChange={(e) => setEditDraft((d) => ({ ...d, why: e.target.value }))}
                          rows={2}
                        />
                        <div className="pause-row">
                          <button className="pill-btn" onClick={() => saveEdit(entry.id)}>Save</button>
                          <button className="link-btn" onClick={() => setEditingId(null)}>Cancel</button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="memory-text">
                          <span className="chip" style={{ color: moodColor(entry.mood_label) }}>
                            {entry.mood_label} · {'●'.repeat(entry.intensity)}{'○'.repeat(5 - entry.intensity)}
                          </span>
                          {entry.time && <span className="hint" style={{ marginRight: 6 }}>{entry.time}</span>}
                          {entry.why}
                          {entry.edited ? <span className="hint"> (edited)</span> : null}
                        </div>
                        <button className="icon-btn" title="Edit" onClick={() => startEdit(entry)} aria-label="Edit entry">
                          <EditIcon />
                        </button>
                        <button className="icon-btn danger" onClick={() => remove(entry.id)} aria-label="Delete entry">
                          <TrashIcon />
                        </button>
                      </>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          ))}
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
function EditIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M3 17.25V21h3.75L17.8 9.94l-3.75-3.75L3 17.25zM20.7 7.04a1 1 0 000-1.41l-2.34-2.34a1 1 0 00-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z" />
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
