"""Test: Does glm-4.6v handle multiple images correctly?"""
import sys, os, base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

# Find test images
img_dir = None
docling_out = Path("docling_output")
if docling_out.exists():
    for d in docling_out.iterdir():
        img_sub = d / "images"
        if img_sub.exists() and list(img_sub.glob("*.png")):
            img_dir = img_sub
            break

if not img_dir:
    # Try backend documents
    for d in Path("backend/documents").iterdir():
        img_sub = d / "images"
        if img_sub.exists() and list(img_sub.glob("*.png")):
            img_dir = img_sub
            break

if not img_dir:
    print("No images found. Process a PDF first.")
    sys.exit(1)

images = sorted(img_dir.glob("*.png"))[:4]
print(f"Testing with {len(images)} images from {img_dir}")

# Build multimodal message with multiple images
content = [{"type": "text", "text": "请简要描述每张图片的内容（每张一句话）。"}]
for img_path in images:
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    content.append({"type": "text", "text": f"\n[{img_path.name}]:"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

print(f"Sending {len(images)} images to glm-4.6v...")
response = client.chat.completions.create(
    model="glm-4.6v",
    messages=[{"role": "user", "content": content}],
    max_tokens=2000,
)

answer = response.choices[0].message.content
tokens = response.usage.total_tokens
reasoning = response.usage.completion_tokens_details.get('reasoning_tokens', 0) if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details else 0

print(f"\n{'='*50}")
print(f"Result: {len(answer)} chars | Tokens: {tokens} (reasoning: {reasoning})")
print(f"{'='*50}")
print(answer[:500])
print(f"\n✅ glm-4.6v handled {len(images)} images successfully!" if answer.strip() else "\n❌ Empty response")
