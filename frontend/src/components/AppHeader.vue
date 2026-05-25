<script setup>
defineProps({ mode: String })
const emit = defineEmits(['update:mode'])
</script>

<template>
  <div class="header-content">
    <div class="header-left">
      <el-icon size="24" color="#409eff"><Document /></el-icon>
      <h1 class="title">MultimodalQA</h1>
      <el-tag type="info" size="small">Document Intelligence Assistant</el-tag>
    </div>
    <div class="header-right">
      <span class="mode-label">RAG Mode:</span>
      <el-switch
        :model-value="mode === 'multimodal'"
        active-text="Multimodal"
        inactive-text="Text Only"
        active-color="#409eff"
        inactive-color="#909399"
        @change="emit('update:mode', $event ? 'multimodal' : 'text_only')"
      />
    </div>
  </div>
</template>

<style scoped>
.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}
.title {
  font-size: 18px;
  margin: 0;
  color: #303133;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.mode-label {
  font-size: 13px;
  color: #606266;
}
</style>
