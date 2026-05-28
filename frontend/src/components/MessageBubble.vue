<script setup>
import { ref } from 'vue'

const props = defineProps({
  message: Object,
  activeDocId: String,
})

const showPagePreview = ref(false)
const previewPageNum = ref(0)
const pagePreviewUrl = ref('')

function getImageSrc(img) {
  if (img.url) return img.url
  if (img.name && props.activeDocId) {
    return `/api/documents/${props.activeDocId}/images/${img.name}`
  }
  return ''
}

function previewPage(pageNum) {
  if (!props.activeDocId || !pageNum) return
  previewPageNum.value = pageNum
  pagePreviewUrl.value = `/api/documents/${props.activeDocId}/pages/${pageNum}`
  showPagePreview.value = true
}
</script>

<template>
  <div class="message-bubble" :class="message.role">
    <!-- Avatar -->
    <div class="avatar">
      <el-icon v-if="message.role === 'user'" size="20"><User /></el-icon>
      <el-icon v-else size="20"><Monitor /></el-icon>
    </div>

    <!-- Content -->
    <div class="bubble-content">
      <!-- Mode badge for assistant -->
      <div v-if="message.role === 'assistant' && message.mode" class="mode-badge">
        <el-tag
          :type="message.mode === 'multimodal' ? '' : 'info'"
          size="small"
          effect="plain"
        >
          {{ message.mode === 'multimodal' ? 'Multimodal RAG' : 'Text-only RAG' }}
        </el-tag>
      </div>

      <!-- Text content -->
      <div class="text-content" v-html="formatContent(message.content)"></div>

      <!-- Inline images (retrieved figures) -->
      <div v-if="message.images && message.images.length" class="image-gallery">
        <div v-for="(img, i) in message.images" :key="i" class="image-item">
          <el-image
            :src="getImageSrc(img)"
            fit="contain"
            :preview-src-list="[getImageSrc(img)]"
            style="max-width: 300px; max-height: 200px; border-radius: 6px;"
          />
          <div class="image-label">{{ img.label }} (Page {{ img.page }})</div>
        </div>
      </div>

      <!-- Citations -->
      <div v-if="message.citations && message.citations.length" class="citations">
        <el-tag
          v-for="(c, i) in message.citations"
          :key="i"
          size="small"
          type="warning"
          effect="plain"
          class="citation-tag"
          @click="previewPage(c.page)"
        >
          📄 Page {{ c.page }}{{ c.label ? ' - ' + c.label : '' }}
        </el-tag>
      </div>

      <!-- Referenced pages (clickable for preview) -->
      <div v-if="message.pages && message.pages.length && message.role === 'assistant'" class="ref-pages">
        <span class="ref-label">Referenced:</span>
        <el-tag
          v-for="p in message.pages"
          :key="p"
          size="small"
          type="info"
          effect="plain"
          class="page-tag"
          @click="previewPage(p)"
        >
          p.{{ p }}
        </el-tag>
      </div>

      <!-- Page preview dialog -->
      <el-dialog
        v-model="showPagePreview"
        :title="`Page ${previewPageNum} Preview`"
        width="700px"
        :append-to-body="true"
      >
        <div class="page-preview-container">
          <el-image
            :src="pagePreviewUrl"
            fit="contain"
            style="width: 100%; max-height: 80vh;"
            :preview-src-list="[pagePreviewUrl]"
          >
            <template #error>
              <div class="page-preview-error">
                <el-icon size="48"><Picture /></el-icon>
                <p>Page image not available</p>
                <p class="hint">Page images are generated during document processing</p>
              </div>
            </template>
          </el-image>
        </div>
      </el-dialog>
    </div>
  </div>
</template>

<script>
export default {
  methods: {
    formatContent(text) {
      if (!text) return ''
      // Convert **bold** to <strong>
      let html = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Convert newlines to <br>
      html = html.replace(/\n/g, '<br>')
      return html
    },
  },
}
</script>

<style scoped>
.message-bubble {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  max-width: 85%;
}
.message-bubble.user {
  margin-left: auto;
  flex-direction: row-reverse;
}
.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.user .avatar {
  background: #409eff;
  color: white;
}
.assistant .avatar {
  background: #e6f7ff;
  color: #409eff;
}
.bubble-content {
  background: #f4f4f5;
  padding: 12px 16px;
  border-radius: 12px;
  max-width: 100%;
}
.user .bubble-content {
  background: #409eff;
  color: white;
  border-radius: 12px 12px 2px 12px;
}
.assistant .bubble-content {
  border-radius: 12px 12px 12px 2px;
}
.mode-badge {
  margin-bottom: 6px;
}
.text-content {
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
}
.image-gallery {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.image-item {
  text-align: center;
}
.image-label {
  font-size: 11px;
  color: #909399;
  margin-top: 4px;
}
.citations {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.citation-tag {
  cursor: pointer;
  transition: transform 0.15s;
}
.citation-tag:hover {
  transform: scale(1.05);
}
.page-tag {
  cursor: pointer;
}
.page-tag:hover {
  color: #409eff;
}
.ref-pages {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.ref-label {
  font-size: 11px;
  color: #909399;
}
.page-preview-container {
  text-align: center;
}
.page-preview-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px;
  color: #909399;
}
.page-preview-error .hint {
  font-size: 12px;
  margin-top: 8px;
}
</style>
