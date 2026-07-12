// Circular profile photo. Static image only (no animation in text mode).
export default function Avatar({ persona, size = 40 }) {
  const src = persona?.photo_url
    ? `${persona.photo_url}?v=${persona._v || 0}`
    : null
  return (
    <div className="avatar" style={{ width: size, height: size }}>
      {src ? (
        <img src={src} alt={persona?.name || 'profile'} draggable="false" />
      ) : (
        <span className="avatar-fallback">{(persona?.name || '?')[0]}</span>
      )}
    </div>
  )
}
