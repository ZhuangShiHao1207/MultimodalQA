import { ref } from 'vue'
import { getDocuments, deleteDocument } from '../api'

/**
 * Document list state management.
 */
export function useDocuments() {
  const documents = ref([])
  const loading = ref(false)

  async function refresh() {
    loading.value = true
    try {
      documents.value = await getDocuments()
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

  // Load on first use
  refresh()

  return { documents, loading, refresh, remove }
}
