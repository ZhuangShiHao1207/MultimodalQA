<script setup>
defineProps({ mode: String })
const emit = defineEmits(['update:mode'])
</script>

<template>
  <div class="header-content">
    <div class="header-left">
      <el-icon size="24" color="#409eff"><Document /></el-icon>
      <h1 class="title">MultimodalQA —— 文档智能助手（多模态RAG）</h1>
    </div>
    <div class="header-right">
      <span class="mode-label">RAG Mode:</span>
      <el-switch
        :model-value="mode"
        active-value="multimodal"
        inactive-value="text_only"
        active-text="Multimodal"
        inactive-text="Text Only"
        active-color="#409eff"
        inactive-color="#909399"
        @update:model-value="emit('update:mode', $event)"
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
