"""
VLM-based summarizer for images and tables.
Uses Zhipu GLM-4V API to generate text descriptions of visual elements.
"""
import os
import base64
import logging
import time
from pathlib import Path
from typing import List, Optional

from zhipuai import ZhipuAI
from dotenv import load_dotenv

from src.ingestion.models import DocumentElement, ElementType

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class VLMSummarizer:
    """
    Generate text summaries for visual document elements using a VLM API.
    These summaries serve as the text representation for vector retrieval.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4v-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Args:
            api_key: Zhipu AI API key (falls back to env var ZHIPUAI_API_KEY)
            model: Model name for multimodal inference
            max_retries: Max retry attempts on API failure
            retry_delay: Seconds between retries
        """
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY not found in environment or parameters")

        self.client = ZhipuAI(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"VLMSummarizer initialized with model: {self.model}")

    def summarize_elements(
        self, elements: List[DocumentElement]
    ) -> List[DocumentElement]:
        """
        Generate summaries for all visual elements (figures and tables).
        Text elements are skipped (they already have text_content).

        Args:
            elements: List of DocumentElements

        Returns:
            Same list with .summary field populated for visual elements
        """
        visual_elements = [
            e for e in elements
            if e.type in (ElementType.FIGURE, ElementType.TABLE)
        ]

        logger.info(f"Generating summaries for {len(visual_elements)} visual elements...")

        for i, elem in enumerate(visual_elements):
            try:
                if elem.type == ElementType.FIGURE:
                    summary = self._summarize_figure(elem)
                elif elem.type == ElementType.TABLE:
                    summary = self._summarize_table(elem)
                else:
                    continue

                elem.summary = summary
                logger.info(
                    f"  [{i+1}/{len(visual_elements)}] {elem.inferred_label} "
                    f"(page {elem.page_number}): {summary[:80]}..."
                )

            except Exception as e:
                logger.error(f"  Failed to summarize {elem.inferred_label}: {e}")
                # Use caption or table text as fallback
                elem.summary = elem.caption or elem.text_content[:500]

        return elements

    def _summarize_figure(self, element: DocumentElement) -> str:
        """Generate a text summary for a figure/image element."""
        if not element.image_path or not Path(element.image_path).exists():
            return element.caption or "图像内容无法获取"

        # Encode image to base64
        image_b64 = self._encode_image(element.image_path)

        prompt = (
            "你是一个专业的文档分析助手。请仔细观察这张图片，生成一段简洁但信息丰富的文本描述。\n"
            "要求：\n"
            "1. 描述图表的类型（折线图/柱状图/散点图/流程图等）\n"
            "2. 概括图表展示的主要趋势、关键数值或核心结论\n"
            "3. 如果有图例/坐标轴标签，请提及关键信息\n"
            "4. 用中文回答，控制在100-200字之间\n"
            "5. 不要说'这张图片显示...'，直接描述内容"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ]

        return self._call_api(messages)

    def _summarize_table(self, element: DocumentElement) -> str:
        """Generate a text summary for a table element."""
        # For tables, we have the markdown representation
        table_md = element.text_content

        if not table_md.strip():
            return "表格内容为空"

        # If table is short, use text-only prompt (faster, no image needed)
        prompt = (
            "你是一个专业的文档分析助手。以下是从文档中提取的一个表格（Markdown格式）。\n"
            "请生成一段简洁的文本摘要，概括表格的核心内容。\n"
            "要求：\n"
            "1. 说明表格包含哪些字段/列\n"
            "2. 总结关键数据对比和趋势（如最高值、最低值、显著差异）\n"
            "3. 用中文回答，控制在100-200字\n"
            "4. 直接描述内容，不要说'这个表格显示...'\n\n"
            f"表格内容：\n{table_md}"
        )

        messages = [{"role": "user", "content": prompt}]

        return self._call_api(messages)

    def _call_api(self, messages: list) -> str:
        """Call the Zhipu API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"    API call failed (attempt {attempt+1}): {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} attempts: {e}")

    def _encode_image(self, image_path: str) -> str:
        """Read and base64-encode an image file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
