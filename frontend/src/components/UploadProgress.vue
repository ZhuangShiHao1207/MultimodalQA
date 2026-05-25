<script setup>
defineProps({
  visible: Boolean,
  stage: String,
  percent: Number,
  message: String,
})

const stageLabels = {
  uploading: 'Uploading',
  parsing: 'Parsing PDF',
  chunking: 'Chunking',
  summarizing: 'VLM Summaries',
  embedding: 'Embedding',
  indexing: 'Building Index',
  done: 'Complete',
  error: 'Error',
}
</script>

<template>
  <el-dialog
    :model-value="visible"
    title="Processing Document"
    width="450px"
    :close-on-click-modal="false"
    :close-on-press-escape="false"
    :show-close="false"
    center
  >
    <div class="progress-content">
      <el-progress
        :percentage="percent"
        :stroke-width="16"
        :text-inside="true"
        striped
        striped-flow
        :status="stage === 'done' ? 'success' : stage === 'error' ? 'exception' : ''"
      />
      <div class="progress-info">
        <el-tag :type="stage === 'error' ? 'danger' : 'primary'" size="small">
          {{ stageLabels[stage] || stage }}
        </el-tag>
        <span class="progress-message">{{ message }}</span>
      </div>
    </div>
  </el-dialog>
</template>

<style scoped>
.progress-content {
  padding: 20px 0;
}
.progress-info {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.progress-message {
  font-size: 13px;
  color: #606266;
}
</style>
