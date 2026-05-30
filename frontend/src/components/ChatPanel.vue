<script setup>
import { ref, nextTick, watch } from 'vue'
import MessageBubble from './MessageBubble.vue'

const props = defineProps({
  messages: Array,
  isStreaming: Boolean,
  disabled: Boolean,
  activeDocId: String,
})
const emit = defineEmits(['send'])

const inputText = ref('')
const chatContainer = ref(null)

// Auto-scroll to bottom on new messages
watch(
  () => props.messages.length,
  async () => {
    await nextTick()
    scrollToBottom()
  }
)

// Also scroll when streaming content updates
watch(
  () => props.messages[props.messages.length - 1]?.content,
  async () => {
    await nextTick()
    scrollToBottom()
  }
)

function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

function handleSend() {
  if (!inputText.value.trim() || props.disabled || props.isStreaming) return
  emit('send', inputText.value.trim())
  inputText.value = ''
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="chat-panel">
    <!-- Messages -->
    <div ref="chatContainer" class="chat-messages">
      <div v-if="!messages.length" class="chat-empty">
        <el-icon size="48" color="#c0c4cc"><ChatDotSquare /></el-icon>
        <p v-if="disabled">请先上传或选择一个文档以开始对话</p>
        <p v-else>对该文档提个问题吧</p>
      </div>

      <MessageBubble
        v-for="(msg, index) in messages"
        :key="index"
        :message="msg"
        :active-doc-id="activeDocId"
      />

      <!-- Streaming indicator -->
      <div v-if="isStreaming" class="streaming-indicator">
        <el-icon class="is-loading"><Loading /></el-icon>
        <span>生成中…</span>
      </div>
    </div>

    <!-- Input Area -->
    <div class="chat-input-area">
      <el-input
        v-model="inputText"
        type="textarea"
        :autosize="{ minRows: 1, maxRows: 4 }"
        :placeholder="disabled ? '请先选择一个文档…' : '对该文档提个问题…'"
        :disabled="disabled || isStreaming"
        @keydown="handleKeydown"
      />
      <el-button
        type="primary"
        :icon="Promotion"
        :disabled="disabled || isStreaming || !inputText.trim()"
        @click="handleSend"
      >
        发送
      </el-button>
    </div>
  </div>
</template>

<script>
import { Promotion } from '@element-plus/icons-vue'
export default { components: { Promotion } }
</script>

<style scoped>
.chat-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.chat-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}
.chat-empty p {
  margin-top: 12px;
  font-size: 14px;
}
.streaming-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  color: #409eff;
  font-size: 13px;
}
.chat-input-area {
  display: flex;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid #e4e7ed;
  background: #fafafa;
  align-items: flex-end;
}
.chat-input-area :deep(.el-textarea__inner) {
  resize: none;
}
</style>
