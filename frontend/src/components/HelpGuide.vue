<script setup>
import { ref } from 'vue'

const visible = ref(false)

function open() {
  visible.value = true
}

defineExpose({ open })
</script>

<template>
  <!-- Floating Help Button -->
  <div class="help-btn" @click="open">
    <el-icon size="22"><QuestionFilled /></el-icon>
  </div>

  <!-- Guide Dialog -->
  <el-dialog
    v-model="visible"
    title="📖 使用指南"
    width="560px"
    :append-to-body="true"
  >
    <div class="guide-content">
      <h3>快速开始</h3>
      <el-steps :active="3" direction="vertical" :space="60">
        <el-step title="上传文档" description="在左侧栏拖拽上传 PDF，或点击上传区域选择文件" />
        <el-step title="构建索引" description="对 pending 状态的文档，点击 ▶ 按钮触发解析与索引构建（约1分钟）" />
        <el-step title="开始问答" description="选中已就绪（✓）的文档，在底部输入框提问即可" />
      </el-steps>

      <el-divider />

      <h3>功能说明</h3>
      <el-descriptions :column="1" border size="small">
        <el-descriptions-item label="模式切换">
          顶部右侧开关可切换 <el-tag size="small">Multimodal</el-tag>（图文联合推理）和
          <el-tag size="small" type="info">Text Only</el-tag>（纯文本基线对比）
        </el-descriptions-item>
        <el-descriptions-item label="文档状态">
          <el-icon color="#909399"><Clock /></el-icon> pending = 未处理 &nbsp;
          <el-icon class="is-loading" color="#e6a23c"><Loading /></el-icon> processing = 处理中 &nbsp;
          <el-icon color="#67c23a"><CircleCheck /></el-icon> ready = 可问答
        </el-descriptions-item>
        <el-descriptions-item label="切换文档">
          点击左侧文档列表中的其他文档即可切换，对话记录会自动清空
        </el-descriptions-item>
        <el-descriptions-item label="引用溯源">
          回答中的 <el-tag size="small" type="warning">Page X</el-tag> 标签标注了答案依据来源的页码
        </el-descriptions-item>
        <el-descriptions-item label="图片预览">
          回答中如果包含检索到的图表，点击图片可放大预览
        </el-descriptions-item>
      </el-descriptions>

      <el-divider />

      <h3>提问技巧</h3>
      <ul class="tips-list">
        <li>🎯 <strong>具体化</strong>：问"Table 1中哪种方法准确率最高"比"结果如何"效果更好</li>
        <li>📊 <strong>图表题</strong>：切到 Multimodal 模式，如"图中损失曲线的收敛趋势是什么"</li>
        <li>📝 <strong>对比题</strong>：如"Random 和 K-Means++ 初始化的区别"</li>
        <li>🔄 <strong>对比模式</strong>：同一问题分别在两种模式下提问，观察答案差异</li>
      </ul>
    </div>
  </el-dialog>
</template>

<style scoped>
.help-btn {
  position: fixed;
  bottom: 24px;
  left: 24px;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: #409eff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.4);
  transition: transform 0.2s, box-shadow 0.2s;
  z-index: 1000;
}
.help-btn:hover {
  transform: scale(1.1);
  box-shadow: 0 6px 16px rgba(64, 158, 255, 0.5);
}
.guide-content h3 {
  margin: 0 0 12px;
  font-size: 15px;
  color: #303133;
}
.tips-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.tips-list li {
  padding: 6px 0;
  font-size: 13px;
  color: #606266;
  line-height: 1.6;
}
</style>
