import { useEffect } from 'react'

// N11: neither overlay panel (Settings, Mood journal) closed on Escape or
// exposed itself as a dialog to assistive tech - a near-universal UX
// expectation for overlay panels, and a real (not stylistic) accessibility
// gap. One small shared hook rather than duplicating the same
// addEventListener/cleanup in both components.
export function useEscapeToClose(onClose) {
  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [onClose])
}
