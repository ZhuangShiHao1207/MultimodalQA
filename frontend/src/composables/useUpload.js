import { ref } from 'vue'
import { uploadPDF, subscribeProgress } from '../api'

/**
 * Upload state management with SSE progress tracking.
 */
export function useUpload() {
  const showProgress = ref(false)
  const progressStage = ref('')
  const progressPercent = ref(0)
  const progressMessage = ref('')

  async function upload(file) {
    showProgress.value = true
    progressStage.value = 'uploading'
    progressPercent.value = 5
    progressMessage.value = 'Uploading file...'

    try {
      // Upload file
      const { task_id, document_id } = await uploadPDF(file)
      progressPercent.value = 10
      progressMessage.value = 'File uploaded, starting processing...'

      // Subscribe to progress events
      return new Promise((resolve, reject) => {
        subscribeProgress(
          task_id,
          (event) => {
            progressStage.value = event.stage
            progressPercent.value = event.progress
            progressMessage.value = event.message

            if (event.stage === 'done') {
              setTimeout(() => {
                showProgress.value = false
              }, 800)
              resolve(document_id)
            } else if (event.stage === 'error') {
              showProgress.value = false
              reject(new Error(event.message))
            }
          },
          (err) => {
            showProgress.value = false
            reject(new Error('Connection lost'))
          }
        )
      })
    } catch (err) {
      showProgress.value = false
      throw err
    }
  }

  return {
    showProgress,
    progressStage,
    progressPercent,
    progressMessage,
    upload,
  }
}
