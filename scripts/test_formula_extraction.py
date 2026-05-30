"""Quick smoke test: re-parse the K-Means/GMM PDF and check formula extraction."""
import sys
import os
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import DoclingParser, ElementType

pdf_path = project_root / "data" / "测试数据（某次实验报告）.pdf"
out_dir = project_root / "scripts" / "_formula_test_out"

parser = DoclingParser(
    output_dir=str(out_dir),
    extract_images=True,
    extract_tables=True,
    generate_page_images=False,  # skip page images for speed
    extract_formulas=True,        # NEW: enable LaTeX OCR
)

print("Parsing... (first run downloads CodeFormulaPredictor weights)")
elements, markdown = parser.parse(str(pdf_path))

# Find formula-bearing elements
formula_elems = [e for e in elements if e.inferred_label == "Formula"]
print(f"\nTotal elements: {len(elements)}")
print(f"Formula-labeled elements: {len(formula_elems)}")

# Count occurrences of LaTeX-looking patterns in markdown
import re
latex_patterns = re.findall(r'\$[^$\n]{2,}\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]', markdown)
print(f"LaTeX-style strings in exported markdown: {len(latex_patterns)}")
print(f"Stale `formula-not-decoded` placeholders: {markdown.count('formula-not-decoded')}")

# Show first few formula samples
if formula_elems:
    print("\n── First 3 formula elements ──")
    for e in formula_elems[:3]:
        print(f"  page={e.page_number} content={e.text_content[:120]!r}")

if latex_patterns:
    print("\n── First 5 LaTeX strings in markdown ──")
    for p in latex_patterns[:5]:
        print(f"  {p[:160]}")

# Also show the K-Means objective area, page 1
kmeans_objective_idx = markdown.find("K-Means 算法的目标函数")
if kmeans_objective_idx >= 0:
    print("\n── K-Means objective context (300 chars after match) ──")
    print(markdown[kmeans_objective_idx:kmeans_objective_idx + 300])
