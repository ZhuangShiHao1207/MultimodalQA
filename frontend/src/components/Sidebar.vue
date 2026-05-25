<script setup>
const props = defineProps({
  documents: Array,
  activeDocId: String,
})
const emit = defineEmits(['select', 'upload', 'delete'])

function handleBeforeUpload(file) {
  emit('upload', file)
  return false // Prevent default upload behavior
}
</script>

<template>
  <div class="sidebar">
    <!-- Upload Button -->
    <div class="upload-section">
      <el-upload
        :before-upload="handleBeforeUpload"
        :show-file-list="false"
        accept=".pdf"
        drag
      >
        <el-icon size="28"><UploadFilled /></el-icon>
        <div class="upload-text">Upload PDF</div>
        <div class="upload-hint">Click or drag file here</div>
      </el-upload>
    </div>

    <!-- Document List -->
    <div class="doc-list-header">
      <span>Documents</span>
      <el-tag size="small" type="info">{{ documents.length }}</el-tag>
    </div>

    <div class="doc-list">
      <div
        v-for="doc in documents"
        :key="doc.id"
        class="doc-item"
        :class="{ active: doc.id === activeDocId }"
        @click="emit('select', doc.id)"
      >
        <div class="doc-info">
          <el-icon v-if="doc.status === 'ready'" color="#67c23a"><CircleCheck /></el-icon>
          <el-icon v-else-if="doc.status === 'processing'" class="is-loading" color="#e6a23c"><Loading /></el-icon>
          <el-icon v-else color="#f56c6c"><CircleClose /></el-icon>
          <span class="doc-name">{{ doc.filename }}</span>
        </div>
        <div class="doc-meta">
          <span v-if="doc.page_count">{{ doc.page_count }} pages</span>
          <el-button
            type="danger"
            size="small"
            text
            @click.stop="emit('delete', doc.id)"
          >
            <el-icon><Delete /></el-icon>
          </el-button>
        </div>
      </div>

      <el-empty v-if="!documents.length" description="No documents yet" :image-size="60" />
    </div>
  </div>
</template>

<style scoped>
.sidebar {
  padding: 16px;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.upload-section {
  margin-bottom: 16px;
}
.upload-section :deep(.el-upload-dragger) {
  padding: 16px;
  height: auto;
}
.upload-text {
  font-size: 14px;
  color: #606266;
  margin-top: 6px;
}
.upload-hint {
  font-size: 12px;
  color: #909399;
}
.doc-list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  font-size: 13px;
  color: #909399;
  font-weight: 600;
}
.doc-list {
  flex: 1;
  overflow-y: auto;
}
.doc-item {
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 4px;
  transition: background 0.2s;
}
.doc-item:hover {
  background: #f5f7fa;
}
.doc-item.active {
  background: #ecf5ff;
  border: 1px solid #b3d8ff;
}
.doc-info {
  display: flex;
  align-items: center;
  gap: 8px;
}
.doc-name {
  font-size: 13px;
  color: #303133;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 180px;
}
.doc-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
  padding-left: 24px;
  font-size: 12px;
  color: #909399;
}
</style>
