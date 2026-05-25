import { ref } from 'vue'
import { streamChat } from '../api'

/**
 * Chat state management.
 * Handles message list, streaming responses, and citation tracking.
 */
export function useChat() {
  const messages = ref([])
  const isStreaming = ref(false)

  async function sendMessage(documentId, question, mode) {
    if (!question.trim() || !documentId) return

    // Add user message
    messages.value.push({
      role: 'user',
      content: question,
      images: [],
      citations: [],
    })

    // Prepare assistant placeholder
    const assistantMsg = {
      role: 'assistant',
      content: '',
      images: [],
      citations: [],
      mode: mode,
      pages: [],
    }
    messages.value.push(assistantMsg)

    isStreaming.value = true

    try {
      let eventCount = 0
      for await (const event of streamChat(documentId, question, mode, [])) {
        eventCount++
        console.log(`[Chat SSE #${eventCount}]`, event.type, event)
        switch (event.type) {
          case 'retrieval':
            assistantMsg.images = event.images || []
            assistantMsg.pages = event.pages || []
            break
          case 'token':
            assistantMsg.content += event.content
            break
          case 'citation':
            assistantMsg.citations.push(event)
            break
          case 'error':
            assistantMsg.content = `Error: ${event.content}`
            break
          case 'done':
            assistantMsg.mode = event.mode || mode
            break
        }
      }
      console.log(`[Chat] Stream ended, received ${eventCount} events`)
      if (eventCount === 0) {
        assistantMsg.content = '⚠️ No response received from server. Please check:\n1. Is the document fully processed (✓ status)?\n2. Check browser F12 console for errors\n3. Check backend terminal for error logs'
      } else if (!assistantMsg.content.trim()) {
        assistantMsg.content = '⚠️ Server returned empty response. Check backend logs for details.'
      }
    } catch (err) {
      console.error('[Chat] Stream error:', err)
      assistantMsg.content = `❌ Request failed: ${err.message}\n\nPlease check backend terminal for detailed error logs.`
    } finally {
      isStreaming.value = false
    }
  }

  function clearChat() {
    messages.value = []
  }

  return { messages, isStreaming, sendMessage, clearChat }
}
