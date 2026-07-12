import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

// Register the service worker for PWA / installability. When a new build is
// deployed, reload once so users never get stuck on a stale cached app.
if ('serviceWorker' in navigator) {
  let refreshing = false
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (refreshing) return
    refreshing = true
    window.location.reload()
  })
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/sw.js')
      .then((reg) => {
        // Check for an updated worker on each load.
        reg.update?.()
        reg.addEventListener?.('updatefound', () => {
          const nw = reg.installing
          nw?.addEventListener('statechange', () => {
            // A new worker is ready while an old one controls the page → activate it.
            if (nw.state === 'installed' && navigator.serviceWorker.controller) {
              nw.postMessage?.({ type: 'SKIP_WAITING' })
            }
          })
        })
      })
      .catch((err) => console.warn('Service worker registration failed:', err))
  })
}
