"""
Grounded multimodal generation with citation support.
Packs retrieved context (text + images) into prompt for GLM-4V.
Generates answers with page/figure citations.
"""
import os
import re
import base64
import logging
from typing import List, Optional

from zhipuai import ZhipuAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


SYSTEM_PROMPT = """你是一个专业的文档问答助手。根据提供的文档内容（包括文本片段、表格和图片），准确回答用户的问题。

**严格要求**：
1. 只根据提供的文档内容回答，不要编造信息
2. 回答结束后，必须标注引用来源，格式为：
   <citation page="页码" type="类型" label="标签"/>
   其中 type 为 text/table/figure，label 为对应标签（如 Table 1, Figure 2）
3. 如果涉及多个来源，列出所有相关引用
4. 如果文档内容无法回答问题，请明确说明"根据提供的文档内容无法回答此问题"
5. 回答要简洁准确，突出关键数据和结论"""

# Third baseline: allows inference when document lacks direct evidence
OPEN_SYSTEM_PROMPT = """你是一个专业的文档问答助手。根据提供的文档内容（包括文本片段、表格和图片），尽力回答用户的问题。

**要求**：
1. 优先根据文档内容直接回答
2. 若文档没有直接信息，可基于文档中的相关上下文进行合理推断，并在回答末尾注明"（推断）"
3. 回答要简洁准确，突出关键数据和结论
4. 即使不确定，也要给出最可能的答案，不要直接拒答"""


class GroundedGenerator:
    """
    Generate answers with citations using retrieved multimodal context.
    Supports text-only and multimodal (text + image) generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4.6v",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY not set")

        self.client = ZhipuAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, context: dict, mode: str = "auto") -> dict:
        """
        Generate a grounded answer from retrieval context.

        Args:
            query:   User's question
            context: Output from MultiVectorRetriever.retrieve_with_context()
            mode:    "auto"      - multimodal if images present, else grounded text-only
                     "grounded"  - text-only, refuses to answer without evidence (default TO baseline)
                     "open"      - text-only, allowed to infer when evidence is indirect

        Returns:
            dict with answer, citations, raw_response, mode
        """
        has_images = bool(context.get("image_contexts"))

        if mode == "auto":
            if has_images:
                response = self._generate_multimodal(query, context)
                used_mode = "multimodal"
            else:
                response = self._generate_text_only(query, context, SYSTEM_PROMPT)
                used_mode = "text_only_grounded"
        elif mode == "grounded":
            response = self._generate_text_only(query, context, SYSTEM_PROMPT)
            used_mode = "text_only_grounded"
        elif mode == "open":
            response = self._generate_text_only(query, context, OPEN_SYSTEM_PROMPT)
            used_mode = "text_only_open"
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'auto', 'grounded', or 'open'.")

        citations = self._parse_citations(response)

        return {
            "answer": self._clean_answer(response),
            "citations": citations,
            "raw_response": response,
            "mode": used_mode,
            "referenced_pages": context.get("all_pages", []),
        }

    def _generate_multimodal(self, query: str, context: dict) -> str:
        """Generate answer using text + images."""
        # Build content array for multimodal message
        content_parts = []

        # Add system instruction + context description
        context_text = self._build_context_text(context)
        content_parts.append({
            "type": "text",
            "text": f"{SYSTEM_PROMPT}\n\n---\n\n**文档上下文：**\n{context_text}\n\n---\n\n**用户问题：** {query}",
        })

        # Add images (glm-4.6v supports multiple images well, limit to 5 for safety)
        max_images_to_send = 5
        for img_ctx in context["image_contexts"][:max_images_to_send]:
            image_b64 = self._encode_image(img_ctx["image_path"])
            content_parts.append({
                "type": "text",
                "text": f"\n[{img_ctx['label']} - 第{img_ctx['page']}页]:",
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            })

        messages = [{"role": "user", "content": content_parts}]

        return self._call_api(messages)

    def _generate_text_only(self, query: str, context: dict, system_prompt: str) -> str:
        """Generate answer using only text context (no images)."""
        context_text = self._build_context_text(context)

        prompt = (
            f"{system_prompt}\n\n---\n\n"
            f"**文档上下文：**\n{context_text}\n\n---\n\n"
            f"**用户问题：** {query}"
        )

        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages)

    def _build_context_text(self, context: dict) -> str:
        """Build text context string from retrieval results."""
        parts = []

        # Text chunks
        for i, tc in enumerate(context.get("text_contexts", []), 1):
            heading = f" [{tc['heading']}]" if tc.get("heading") else ""
            parts.append(
                f"**文本片段{i}** (第{tc['page']}页{heading}):\n{tc['content']}\n"
            )

        # Tables
        for tc in context.get("table_contexts", []):
            parts.append(
                f"**{tc['label']}** (第{tc['page']}页):\n{tc['content']}\n"
            )

        # Image descriptions (text form)
        for ic in context.get("image_contexts", []):
            parts.append(
                f"**{ic['label']}** (第{ic['page']}页) - 摘要: {ic['summary']}\n"
            )

        return "\n".join(parts) if parts else "（无相关文档内容）"

    def _call_api(self, messages: list) -> str:
        """Call Zhipu API and clean response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw = response.choices[0].message.content.strip()
            # Clean GLM special tokens that leak in multi-image scenarios
            cleaned = self._clean_special_tokens(raw)
            return cleaned
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"生成失败: {e}"

    def _clean_special_tokens(self, text: str) -> str:
        """Remove GLM model special tokens that shouldn't appear in output."""
        # GLM sometimes outputs these when confused (especially with multiple images)
        special_tokens = [
            "<|observation|>", "<|user|>", "<|assistant|>", "<|system|>",
            "<|endoftext|>", "<|tool|>", "<|result|>",
        ]
        for token in special_tokens:
            text = text.replace(token, "")

        # If after cleaning the text is empty, it means the model failed to generate
        text = text.strip()
        if not text:
            return "模型未能生成有效回答，请尝试简化问题或减少问题中涉及的图片数量。"

        return text

    def _parse_citations(self, response: str) -> List[dict]:
        """Extract citation tags from response."""
        pattern = r'<citation\s+page="(\d+)"\s+type="(\w+)"\s+label="([^"]*)"'
        matches = re.findall(pattern, response)

        citations = []
        for page, ctype, label in matches:
            citations.append({
                "page": int(page),
                "type": ctype,
                "label": label,
            })

        # Also try to extract page references from natural language
        if not citations:
            page_pattern = r'第\s*(\d+)\s*页'
            page_matches = re.findall(page_pattern, response)
            for p in set(page_matches):
                citations.append({"page": int(p), "type": "inferred", "label": ""})

        return citations

    def _clean_answer(self, response: str) -> str:
        """Remove citation XML tags from display text."""
        cleaned = re.sub(r'<citation[^/]*/>', '', response)
        return cleaned.strip()

    def _encode_image(self, image_path: str) -> str:
        """Base64-encode an image file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
