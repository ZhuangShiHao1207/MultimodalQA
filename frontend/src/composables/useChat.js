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
      for await (const event of streamChat(documentId, question, mode, [])) {
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
    } catch (err) {
      assistantMsg.content = `Request failed: ${err.message}`
    } finally {
      isStreaming.value = false
    }
  }

  function clearChat() {
    messages.value = []
  }

  return { messages, isStreaming, sendMessage, clearChat }
}
