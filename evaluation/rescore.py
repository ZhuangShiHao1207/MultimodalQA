"""
Recompute ANLS scores from saved results.json using the updated metrics.
This avoids re-querying the LLM (which costs API calls).
"""
import sys
import json
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.metrics import anls_score

results_path = project_root / "evaluation" / "results.json"
with open(results_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

results = summary["details"]
print(f"Re-scoring {len(results)} questions with updated metric...\n")

# Recompute per-question scores
for r in results:
    r["multimodal_anls"] = anls_score(r["multimodal_answer"], r["gold_answers"])
    r["text_only_anls"] = anls_score(r["text_only_answer"], r["gold_answers"])
    r["mm_correct"] = r["multimodal_anls"] >= 0.5
    r["to_correct"] = r["text_only_anls"] >= 0.5

# Aggregate
n = len(results)
mm_avg = sum(r["multimodal_anls"] for r in results) / n
to_avg = sum(r["text_only_anls"] for r in results) / n
mm_acc = sum(1 for r in results if r["mm_correct"]) / n
to_acc = sum(1 for r in results if r["to_correct"]) / n

# By type
types = sorted(set(r["type"] for r in results))
type_metrics = {}
for t in types:
    tr = [r for r in results if r["type"] == t]
    type_metrics[t] = {
        "count": len(tr),
        "multimodal_anls": sum(r["multimodal_anls"] for r in tr) / len(tr),
        "text_only_anls": sum(r["text_only_anls"] for r in tr) / len(tr),
        "mm_accuracy": sum(1 for r in tr if r["mm_correct"]) / len(tr),
        "to_accuracy": sum(1 for r in tr if r["to_correct"]) / len(tr),
    }

# Visual vs non-visual
visual = [r for r in results if r["requires_visual"]]
non_visual = [r for r in results if not r["requires_visual"]]

summary["overall"] = {
    "multimodal_anls": mm_avg,
    "text_only_anls": to_avg,
    "multimodal_accuracy": mm_acc,
    "text_only_accuracy": to_acc,
    "anls_improvement": mm_avg - to_avg,
    "accuracy_improvement": mm_acc - to_acc,
}
summary["by_type"] = type_metrics
summary["visual_questions"] = {
    "count": len(visual),
    "multimodal_anls": sum(r["multimodal_anls"] for r in visual) / max(len(visual), 1),
    "text_only_anls": sum(r["text_only_anls"] for r in visual) / max(len(visual), 1),
    "mm_accuracy": sum(1 for r in visual if r["mm_correct"]) / max(len(visual), 1),
    "to_accuracy": sum(1 for r in visual if r["to_correct"]) / max(len(visual), 1),
}
summary["non_visual_questions"] = {
    "count": len(non_visual),
    "multimodal_anls": sum(r["multimodal_anls"] for r in non_visual) / max(len(non_visual), 1),
    "text_only_anls": sum(r["text_only_anls"] for r in non_visual) / max(len(non_visual), 1),
    "mm_accuracy": sum(1 for r in non_visual if r["mm_correct"]) / max(len(non_visual), 1),
    "to_accuracy": sum(1 for r in non_visual if r["to_correct"]) / max(len(non_visual), 1),
}

# Print summary
print("=" * 70)
print("RECOMPUTED EVALUATION RESULTS (with substring/numeric matching)")
print("=" * 70)
print(f"\nOverall:")
print(f"  Multimodal RAG: ANLS={mm_avg:.4f}  Acc={mm_acc:.2%}")
print(f"  Text-only  RAG: ANLS={to_avg:.4f}  Acc={to_acc:.2%}")
print(f"  Improvement:    +{mm_avg - to_avg:.4f} ANLS, +{(mm_acc - to_acc)*100:.1f}pp Acc")

print(f"\nBy Type:")
for t, m in type_metrics.items():
    print(f"  {t:8s} ({m['count']:2d}): MM ANLS={m['multimodal_anls']:.3f}/Acc={m['mm_accuracy']:.0%}  TO ANLS={m['text_only_anls']:.3f}/Acc={m['to_accuracy']:.0%}")

v = summary['visual_questions']
nv = summary['non_visual_questions']
print(f"\nVisual    ({v['count']:2d}): MM ANLS={v['multimodal_anls']:.3f}/Acc={v['mm_accuracy']:.0%}  TO ANLS={v['text_only_anls']:.3f}/Acc={v['to_accuracy']:.0%}")
print(f"NonVisual ({nv['count']:2d}): MM ANLS={nv['multimodal_anls']:.3f}/Acc={nv['mm_accuracy']:.0%}  TO ANLS={nv['text_only_anls']:.3f}/Acc={nv['to_accuracy']:.0%}")

# Print per-question breakdown
print(f"\nPer-Question Results:")
print(f"{'ID':4s} {'Type':8s} {'Vis':4s} {'MM':6s} {'TO':6s} Question")
for r in results:
    vis = '[V]' if r["requires_visual"] else '   '
    mm_mark = 'OK' if r["mm_correct"] else 'NG'
    to_mark = 'OK' if r["to_correct"] else 'NG'
    print(f"{r['id']:4s} {r['type']:8s} {vis:4s} MM={mm_mark}({r['multimodal_anls']:.2f}) TO={to_mark}({r['text_only_anls']:.2f})  {r['question'][:60]}")

# Mark MM failures (key info for the report)
mm_failures = [r for r in results if not r["mm_correct"]]
print(f"\n{'='*70}\nMULTIMODAL RAG FAILURES ({len(mm_failures)}/{n}):\n{'='*70}")
for f in mm_failures:
    print(f"\n[{f['id']}] {f['type']} (visual={f['requires_visual']})")
    print(f"  Q: {f['question']}")
    print(f"  Gold: {f['gold_answers']}")
    print(f"  Got:  {f['multimodal_answer'][:300]}")

# Save updated results
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nUpdated results saved to: {results_path}")
