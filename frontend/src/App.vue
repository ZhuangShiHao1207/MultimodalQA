<script setup>
import { ref } from 'vue'
import AppHeader from './components/AppHeader.vue'
import Sidebar from './components/Sidebar.vue'
import ChatPanel from './components/ChatPanel.vue'
import UploadProgress from './components/UploadProgress.vue'
import { useChat } from './composables/useChat'
import { useUpload } from './composables/useUpload'
import { useDocuments } from './composables/useDocuments'

// State
const mode = ref('multimodal')
const activeDocId = ref(null)

// Composables
const { messages, isStreaming, sendMessage, clearChat } = useChat()
const { documents, refresh: refreshDocs, remove: removeDocs } = useDocuments()
const { showProgress, progressStage, progressPercent, progressMessage, upload } = useUpload()

// Handlers
async function handleUpload(file) {
  try {
    const docId = await upload(file)
    await refreshDocs()
    activeDocId.value = docId
    clearChat()
  } catch (err) {
    console.error('Upload failed:', err)
  }
}

function handleSelectDoc(docId) {
  activeDocId.value = docId
  clearChat()
}

async function handleDeleteDoc(docId) {
  await removeDocs(docId)
  if (activeDocId.value === docId) {
    activeDocId.value = null
    clearChat()
  }
}

function handleSend(question) {
  if (!activeDocId.value) return
  sendMessage(activeDocId.value, question, mode.value)
}
</script>

<template>
  <el-container class="app-container">
    <!-- Top Header -->
    <el-header class="app-header">
      <AppHeader v-model:mode="mode" />
    </el-header>

    <el-container class="app-body">
      <!-- Left Sidebar -->
      <el-aside width="280px" class="app-sidebar">
        <Sidebar
          :documents="documents"
          :active-doc-id="activeDocId"
          @select="handleSelectDoc"
          @upload="handleUpload"
          @delete="handleDeleteDoc"
        />
      </el-aside>

      <!-- Main Chat Area -->
      <el-main class="app-main">
        <ChatPanel
          :messages="messages"
          :is-streaming="isStreaming"
          :disabled="!activeDocId"
          :active-doc-id="activeDocId"
          @send="handleSend"
        />
      </el-main>
    </el-container>

    <!-- Upload Progress Overlay -->
    <UploadProgress
      :visible="showProgress"
      :stage="progressStage"
      :percent="progressPercent"
      :message="progressMessage"
    />
  </el-container>
</template>

<style>
html, body, #app {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
</style>

<style scoped>
.app-container {
  height: 100vh;
}
.app-header {
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  padding: 0 20px;
  height: 56px;
}
.app-body {
  height: calc(100vh - 56px);
}
.app-sidebar {
  border-right: 1px solid #e4e7ed;
  overflow-y: auto;
}
.app-main {
  padding: 0;
  overflow: hidden;
}
</style>
