// Minimal service worker for installability + offline app shell.
// Only known-static paths are ever cached; every other same-origin request
// (all API endpoints) goes straight to the network. The old version kept a
// blocklist of API prefixes instead — /relationship, /journal, /day-state
// etc. weren't on it, so their FIRST response was cached and served stale
// forever (the "stats never update / journal looks empty" bug).

const CACHE = 'companion-shell-v4'
const SHELL = ['/', '/index.html', '/manifest.json', '/icons/icon-192.png']

// Truly static, safe-to-cache paths (hashed bundles, icons, VAD wasm...).
const STATIC_PREFIXES = ['/assets/', '/icons/', '/avatars/', '/vad/']
const STATIC_FILES = ['/manifest.json', '/favicon.ico']

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).catch(() => {}))
  self.skipWaiting()
})

// Let the page tell a waiting worker to take over immediately.
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  )
  self.clients.claim()
})

self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)

  // Only handle same-origin GETs; everything else (POST /chat, etc.) passes through.
  if (request.method !== 'GET' || url.origin !== self.location.origin) return

  // Navigations (HTML) are network-first so a new build is picked up immediately
  // and users never get stuck on a stale app shell.
  if (request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html')) {
    event.respondWith(
      fetch(request)
        .then((res) => {
          const copy = res.clone()
          caches.open(CACHE).then((c) => c.put(request, copy)).catch(() => {})
          return res
        })
        .catch(() => caches.match(request).then((c) => c || caches.match('/index.html')))
    )
    return
  }

  const isStatic =
    STATIC_PREFIXES.some((p) => url.pathname.startsWith(p)) ||
    STATIC_FILES.includes(url.pathname)
  if (!isStatic) return // live data (API) — never intercepted, never cached

  // Hashed static assets: cache-first (they're immutable per build).
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached
      return fetch(request)
        .then((res) => {
          const copy = res.clone()
          caches.open(CACHE).then((c) => c.put(request, copy)).catch(() => {})
          return res
        })
        .catch(() => caches.match('/index.html'))
    })
  )
})
