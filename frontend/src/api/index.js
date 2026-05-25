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

// ─── SSE: Chat Streaming ─────────────────────────────────────

export async function* streamChat(documentId, question, mode, history = []) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, question, mode, history }),
  })

  if (!res.ok) {
    throw new Error(`Chat failed: ${res.statusText}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Parse SSE format: "data: {...}\n\n"
    const parts = buffer.split('\n\n')
    buffer = parts.pop() // Keep incomplete last part

    for (const part of parts) {
      const match = part.match(/^data:\s*(.+)$/m)
      if (match) {
        try {
          yield JSON.parse(match[1])
        } catch (e) {
          console.warn('Failed to parse SSE event:', match[1])
        }
      }
    }
  }
}
