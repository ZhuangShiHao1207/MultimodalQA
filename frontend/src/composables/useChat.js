import { ref, reactive } from 'vue'
import { sendChatMessage } from '../api'

/**
 * Chat state management.
 * Uses direct JSON response + frontend typing animation.
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

    // Add assistant placeholder as reactive object
    messages.value.push({
      role: 'assistant',
      content: '',
      images: [],
      citations: [],
      mode: mode,
      pages: [],
    })

    // Get reference to the REACTIVE proxy in the array (critical for Vue reactivity!)
    const assistantMsg = messages.value[messages.value.length - 1]

    isStreaming.value = true

    try {
      // Send request and get complete response
      const result = await sendChatMessage(documentId, question, mode, [])
      console.log('[Chat] Response received:', result)

      // Set metadata immediately
      assistantMsg.images = result.images || []
      assistantMsg.citations = result.citations || []
      assistantMsg.pages = result.pages || []
      assistantMsg.mode = result.mode || mode

      // Get the full answer
      const fullAnswer = result.answer || ''
      if (!fullAnswer) {
        assistantMsg.content = '⚠️ Empty response from server.'
        return
      }

      // Typing animation: reveal answer progressively
      const chunkSize = 8
      for (let i = 0; i < fullAnswer.length; i += chunkSize) {
        assistantMsg.content = fullAnswer.substring(0, i + chunkSize)
        await sleep(20)
      }
      // Ensure full content is shown
      assistantMsg.content = fullAnswer

    } catch (err) {
      console.error('[Chat] Error:', err)
      assistantMsg.content = `❌ Request failed: ${err.message}`
    } finally {
      isStreaming.value = false
    }
  }

  function clearChat() {
    messages.value = []
  }

  return { messages, isStreaming, sendMessage, clearChat }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
