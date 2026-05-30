import { ref, computed } from 'vue'
import { sendChatMessage } from '../api'

/**
 * Chat state management with per-document history.
 * Each document maintains its own conversation.
 */
export function useChat() {
  // Per-document chat histories: { docId: [messages] }
  const chatHistories = ref({})
  const activeDocId = ref(null)
  const isStreaming = ref(false)

  // Current document's messages (reactive computed)
  const messages = computed(() => {
    if (!activeDocId.value) return []
    return chatHistories.value[activeDocId.value] || []
  })

  function setActiveDoc(docId) {
    activeDocId.value = docId
    // Initialize history for new doc if needed
    if (docId && !chatHistories.value[docId]) {
      chatHistories.value[docId] = []
    }
  }

  async function sendMessage(documentId, question, mode) {
    if (!question.trim() || !documentId) return

    // Ensure history array exists
    if (!chatHistories.value[documentId]) {
      chatHistories.value[documentId] = []
    }
    const history = chatHistories.value[documentId]

    // Add user message
    history.push({
      role: 'user',
      content: question,
      images: [],
      citations: [],
    })

    // Add assistant placeholder
    history.push({
      role: 'assistant',
      content: '',
      images: [],
      citations: [],
      mode: mode,
      pages: [],
    })

    // Get reactive reference to the assistant message
    const assistantMsg = history[history.length - 1]

    isStreaming.value = true

    try {
      const result = await sendChatMessage(documentId, question, mode, [])
      console.log('[Chat] Response received:', result)

      // Set metadata
      assistantMsg.images = result.images || []
      assistantMsg.citations = result.citations || []
      assistantMsg.pages = result.pages || []
      // Prefer the user-requested mode for the badge display. Backend now
      // returns this in `mode`, but fall back to the local `mode` arg if a
      // proxy strips the field.
      assistantMsg.mode = result.mode || mode

      // Typing animation
      const fullAnswer = result.answer || ''
      if (!fullAnswer) {
        assistantMsg.content = '⚠️ Empty response from server.'
        return
      }

      const chunkSize = 8
      for (let i = 0; i < fullAnswer.length; i += chunkSize) {
        assistantMsg.content = fullAnswer.substring(0, i + chunkSize)
        await sleep(20)
      }
      assistantMsg.content = fullAnswer

    } catch (err) {
      console.error('[Chat] Error:', err)
      assistantMsg.content = `❌ Request failed: ${err.message}`
    } finally {
      isStreaming.value = false
    }
  }

  function clearChat(docId) {
    const id = docId || activeDocId.value
    if (id) {
      chatHistories.value[id] = []
    }
  }

  function clearAllChats() {
    chatHistories.value = {}
  }

  return { messages, isStreaming, sendMessage, clearChat, clearAllChats, setActiveDoc }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
