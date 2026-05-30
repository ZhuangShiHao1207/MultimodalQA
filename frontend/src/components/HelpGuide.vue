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
    width="640px"
    :append-to-body="true"
  >
    <div class="guide-content">
      <!-- ── 系统能力 ── -->
      <h3>🚀 系统能做什么</h3>
      <ul class="capability-list">
        <li>
          <strong>多模态文档解析</strong>：自动从 PDF 抽取文本、表格、图表（柱状图/混淆矩阵/训练曲线/散点图等）以及
          <strong>数学公式</strong>（OCR 成 LaTeX）
        </li>
        <li>
          <strong>带溯源的问答</strong>：回答附带页码与图表标签，点击 📄 标签可弹窗预览原始页面
        </li>
        <li>
          <strong>多模态 vs 纯文本对比</strong>：右上角一键切换 RAG 模式，体验图像信息对图表类问题的提升
        </li>
        <li>
          <strong>LaTeX 公式渲染</strong>：回答中的公式（如 <code>\(\sum x_i\)</code>、<code>$$E=mc^2$$</code>）会用 KaTeX 渲染为漂亮的数学符号
        </li>
        <li>
          <strong>持久化索引</strong>：ChromaDB 把向量写入磁盘，重启后端无需重新解析已处理过的文档
        </li>
      </ul>

      <el-divider />

      <!-- ── 快速开始 ── -->
      <h3>⚡ 快速开始</h3>
      <el-steps :active="3" direction="vertical" :space="64">
        <el-step
          title="上传或放置 PDF"
          description="左侧栏拖拽上传，或直接把 PDF 放到项目 data/ 目录后重启后端（约 2 分钟自动检测）。建议页数 < 10 的小 PDF 测试更快。"
        />
        <el-step
          title="构建索引"
          description="对 ⏰ pending 状态的文档点 ▶ 触发处理。整体 2~5 分钟，PDF 页数和图片越多越久。首次会下载 Docling/BGE-M3/公式 OCR 模型权重。"
        />
        <el-step
          title="开始问答"
          description="选中 ✓ ready 文档，输入问题。每个文档独立保存对话历史。"
        />
      </el-steps>

      <el-divider />

      <!-- ── 功能说明 ── -->
      <h3>🛠 功能说明</h3>
      <el-descriptions :column="1" border size="small">
        <el-descriptions-item label="模式切换">
          顶部右侧开关切换
          <el-tag size="small">Multimodal</el-tag>（图文联合推理，看图回答）和
          <el-tag size="small" type="info">Text Only</el-tag>（消融基线，仅文本）
        </el-descriptions-item>
        <el-descriptions-item label="文档状态">
          <el-icon color="#909399"><Clock /></el-icon> pending = 待处理 &nbsp;
          <el-icon class="is-loading" color="#e6a23c"><Loading /></el-icon> processing = 处理中 &nbsp;
          <el-icon color="#67c23a"><CircleCheck /></el-icon> ready = 可问答
        </el-descriptions-item>
        <el-descriptions-item label="切换文档">
          点左侧文档列表切换，<strong>每个文档独立保存对话历史</strong>，不会丢
        </el-descriptions-item>
        <el-descriptions-item label="引用溯源">
          回答中的
          <el-tag size="small" type="warning">📄 第 X 页</el-tag>
          标签可点击弹窗预览原始页面
        </el-descriptions-item>
        <el-descriptions-item label="图片放大">
          回答附带的检索图片，点击可放大全屏预览
        </el-descriptions-item>
        <el-descriptions-item label="公式渲染">
          回答中的 LaTeX（<code>\(...\)</code> / <code>\[...\]</code> / <code>$$...$$</code> / 含 <code>\^_{ }</code> 的 <code>$...$</code>）会自动渲染
        </el-descriptions-item>
      </el-descriptions>

      <el-divider />

      <!-- ── 演示推荐 ── -->
      <h3>🎬 演示推荐场景</h3>
      <ul class="tips-list">
        <li>
          📊 <strong>图表趋势题</strong>（多模态优势最大）：
          切到 Multimodal 模式，问"从训练损失曲线来看，哪种初始化损失下降更快？"
          → 切到 Text Only 同问 → 对比答案差异
        </li>
        <li>
          🔢 <strong>混淆矩阵单元格题</strong>：
          "TF-IDF + LogisticRegression 的混淆矩阵中，true label 为 negative 但预测为 positive 的样本数？"
          （只有看图才能答对）
        </li>
        <li>
          📐 <strong>公式提取题</strong>：
          在 K-Means/GMM 这类含公式的文档上，问"K-Means 的目标函数是什么？请把公式写出来"
          → 前端会以 KaTeX 渲染数学符号
        </li>
        <li>
          🔗 <strong>引用溯源</strong>：
          任何回答下方都有"引用页码"标签，点一下弹出原页预览，验证答案的真实性
        </li>
        <li>
          🔄 <strong>消融对比</strong>：
          同一题在 Multimodal 和 Text Only 下分别问，观察纯文本因缺图像而拒答或答错的现象
        </li>
      </ul>

      <el-divider />

      <!-- ── 提问技巧 ── -->
      <h3>💡 提问技巧</h3>
      <ul class="tips-list">
        <li>🎯 <strong>具体化</strong>：问"Table 1 中哪种方法准确率最高"比"结果如何"效果更好</li>
        <li>📊 <strong>引用图标</strong>：明确提到"从…图来看"会引导模型走视觉路径</li>
        <li>📝 <strong>对比题</strong>：如"Random 和 K-Means++ 初始化的区别"</li>
        <li>🚫 <strong>无关问题</strong>：模型会拒答（"根据提供的文档内容无法回答"），这是设计行为，不是 Bug</li>
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
.capability-list,
.tips-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.capability-list li,
.tips-list li {
  padding: 6px 0;
  font-size: 13px;
  color: #606266;
  line-height: 1.7;
}
.capability-list strong,
.tips-list strong {
  color: #303133;
}
code {
  background: #f0f2f5;
  padding: 1px 5px;
  border-radius: 3px;
  font-family: ui-monospace, "Consolas", monospace;
  font-size: 0.92em;
  color: #d63384;
}
</style>
