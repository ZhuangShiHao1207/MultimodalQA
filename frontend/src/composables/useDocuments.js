import { ref, onMounted, onUnmounted } from 'vue'
import { getDocuments, deleteDocument } from '../api'

/**
 * Document list state management.
 * Auto-polls every 5s until documents are found.
 */
export function useDocuments() {
  const documents = ref([])
  const loading = ref(false)
  let pollTimer = null

  async function refresh() {
    loading.value = true
    try {
      documents.value = await getDocuments()
      // Stop polling once we have documents
      if (documents.value.length > 0 && pollTimer) {
        clearInterval(pollTimer)
        pollTimer = null
      }
    } catch (err) {
      console.error('Failed to load documents:', err)
    } finally {
      loading.value = false
    }
  }

  async function remove(docId) {
    await deleteDocument(docId)
    documents.value = documents.value.filter((d) => d.id !== docId)
  }

  // Load on first use + poll every 5s until docs appear
  refresh()
  pollTimer = setInterval(() => {
    if (documents.value.length === 0) {
      refresh()
    }
  }, 5000)

  return { documents, loading, refresh, remove }
}
