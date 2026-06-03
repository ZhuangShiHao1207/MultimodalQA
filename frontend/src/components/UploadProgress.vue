<script setup>
import { computed } from 'vue'

const props = defineProps({
  visible: Boolean,
  stage: String,
  percent: Number,
  message: String,
})

const stageLabels = {
  uploading: '上传中',
  parsing: '解析 PDF',
  chunking: '文本切分',
  summarizing: '生成图表摘要',
  embedding: '向量化索引',
  indexing: '构建索引',
  done: '完成',
  error: '错误',
}

// Show a friendly time-estimate hint that depends on the current stage.
const stageHint = computed(() => {
  switch (props.stage) {
    case 'uploading':
      return '正在上传文件…'
    case 'parsing':
      return '正在用 Docling 做版面分析（最耗时阶段，约3~5分钟），PDF 页数/图片越多耗时越久。'
    case 'chunking':
      return '正在按语义切分文本…'
    case 'summarizing':
      return '调用 GLM-4.6V 为每张图表/表格生成可检索摘要，每个视觉元素约 5~10 秒。'
    case 'embedding':
      return '使用 BGE-M3 向量化并写入 ChromaDB…'
    case 'done':
      return '索引构建完成，可以开始问答了！'
    case 'error':
      return '处理失败，请查看后端日志或重试。'
    default:
      return '请耐心等待，若是首次启动会下载模型权重，除开下载时间整个文本处理流程通常需要4~7分钟。'
  }
})
</script>

<template>
  <el-dialog
    :model-value="visible"
    title="📄 文档处理中"
    width="480px"
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

      <!-- Friendly time estimate hint -->
      <div
        v-if="stage !== 'done' && stage !== 'error'"
        class="progress-hint"
      >
        <el-icon><InfoFilled /></el-icon>
        <span>{{ stageHint }}</span>
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
.progress-hint {
  margin-top: 14px;
  padding: 10px 12px;
  background: #f4f7fb;
  border-left: 3px solid #409eff;
  border-radius: 4px;
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 12px;
  line-height: 1.6;
  color: #606266;
}
.progress-hint .el-icon {
  color: #409eff;
  flex-shrink: 0;
  margin-top: 2px;
}
</style>
