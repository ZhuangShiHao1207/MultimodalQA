/**
 * API layer - all backend communication in one file.
 * Endpoints proxy through Vite to http://localhost:8000
 */

// ─── REST Calls ──────────────────────────────────────────────

export async function uploadPDF(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/upload', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`)
  return res.json() // { task_id, document_id }
}

export async function getDocuments() {
  const res = await fetch('/api/documents')
  return res.json()
}

export async function deleteDocument(docId) {
  await fetch(`/api/documents/${docId}`, { method: 'DELETE' })
}

export function getImageUrl(docId, imgName) {
  return `/api/documents/${docId}/images/${imgName}`
}

// ─── SSE: Upload Progress ────────────────────────────────────

export function subscribeProgress(taskId, onEvent, onError) {
  const es = new EventSource(`/api/upload/${taskId}/progress`)
  es.onmessage = (e) => {
    const data = JSON.parse(e.data)
    onEvent(data)
    if (data.stage === 'done' || data.stage === 'error') {
      es.close()
    }
  }
  es.onerror = (e) => {
    es.close()
    if (onError) onError(e)
  }
  return es
}

// ─── Chat: Direct JSON (no SSE) ──────────────────────────────

export async function sendChatMessage(documentId, question, mode, history = []) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, question, mode, history }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Chat failed (${res.status}): ${text}`)
  }

  return res.json()
  // Returns: { answer, images, citations, pages, mode, error }
}
