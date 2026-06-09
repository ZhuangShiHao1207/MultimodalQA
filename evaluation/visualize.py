"""
evaluation/visualize.py
Generate evaluation charts from results.json.
Produces 6 figures saved to evaluation/figures/:
  fig1_main_comparison.png  - MM vs TO-Grounded vs TO-Open by question type (3-way bar)
  fig2_visual_split.png     - 3-way: visual vs non-visual split
  fig3_per_question.png     - Per-question ANLS heatmap (2 or 3 rows depending on data)
  fig4_to_failure_pie.png   - Text-only Grounded failure mode breakdown (pie)
  fig5_mcnemar.png          - McNemar test result summary table (MM vs TO)
  fig7_difficulty.png       - Accuracy by difficulty level (3-way bar, if available)

Run with:
  python3 -m evaluation.visualize          (from project root)
  python3 evaluation/visualize.py          (direct)
"""

import json
import sys
import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless / no-display safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── font: prefer a CJK-capable font on each platform ──────────────────────────
def _setup_fonts():
    candidates = [
        "PingFang SC", "Heiti SC", "STHeiti",        # macOS
        "Microsoft YaHei", "SimHei", "FangSong",      # Windows
        "WenQuanYi Micro Hei", "Noto Sans CJK SC",    # Linux
        "Arial Unicode MS",
    ]
    from matplotlib import font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            matplotlib.rcParams["font.family"] = c
            return
    # fallback – use DejaVu and replace CJK with ASCII equivalents in labels
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

_setup_fonts()
matplotlib.rcParams["axes.unicode_minus"] = False


# ── paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
RESULTS_PATH = _HERE / "results.json"
OUT_DIR = _HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "mm":   "#4C72B0",   # blue    – Multimodal RAG
    "to":   "#DD8452",   # orange  – Text-only Grounded
    "open": "#55A868",   # green   – Text-only Open
    "ok":   "#55A868",
    "err":  "#C44E52",   # red
    "mid":  "#8172B2",   # purple
}

def _has_open(data: dict) -> bool:
    """Check whether results.json contains text_open scores (3-way eval)."""
    overall = data.get("overall", {})
    return "text_open_accuracy" in overall


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Main comparison bar chart (by question type) — 2-way or 3-way
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_main_comparison(data: dict, out_dir: Path):
    by_type = data["by_type"]
    overall = data["overall"]
    three   = _has_open(data)

    type_labels = {
        "figure": "figure\n(视觉图表)",
        "table":  "table\n(表格数据)",
        "text":   "text\n(纯文本)",
    }
    type_keys  = [k for k in ["figure", "table", "text"] if k in by_type]
    categories = [type_labels.get(t, t) for t in type_keys] + ["整体\n(Overall)"]
    counts     = [by_type[t]["count"] for t in type_keys] + [data["total_questions"]]

    mm_acc   = [by_type[t]["mm_accuracy"]   for t in type_keys] + [overall["multimodal_accuracy"]]
    to_acc   = [by_type[t]["to_accuracy"]   for t in type_keys] + [overall["text_only_accuracy"]]
    open_acc = ([by_type[t].get("open_accuracy", 0) for t in type_keys]
                + [overall.get("text_open_accuracy", 0)]) if three else None

    x   = np.arange(len(categories))
    n_bars = 3 if three else 2
    w   = 0.25 if three else 0.35
    offsets = ([-w, 0, w] if three else [-w/2, w/2])

    fig, ax = plt.subplots(figsize=(11 if three else 10, 5.5))

    bars = []
    for idx, (vals, label, color) in enumerate([
        (mm_acc,   "Multimodal RAG",        COLORS["mm"]),
        (to_acc,   "Text-only Grounded",    COLORS["to"]),
        *( [(open_acc, "Text-only Open", COLORS["open"])] if three else [] ),
    ]):
        b = ax.bar(x + offsets[idx], vals, w, label=label, color=color, alpha=0.88, zorder=3)
        bars.append(b)
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, (n, _) in enumerate(zip(counts, categories)):
        ax.text(i, -0.08, f"n={n}", ha="center", va="top", fontsize=9,
                color="gray", transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    title = ("三路对比  —  按题目类型" if three else
             "Multimodal RAG vs Text-only RAG  —  按题目类型对比")
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "fig1_main_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 – Visual vs Non-visual split — 2-way or 3-way
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_visual_split(data: dict, out_dir: Path):
    vq    = data["visual_questions"]
    nvq   = data["non_visual_questions"]
    three = _has_open(data)

    labels = [
        f"视觉题\n(Visual, n={vq['count']})",
        f"非视觉题\n(Non-visual, n={nvq['count']})",
    ]
    mm_vals   = [vq["mm_accuracy"],           nvq["mm_accuracy"]]
    to_vals   = [vq["to_accuracy"],           nvq["to_accuracy"]]
    open_vals = ([vq.get("open_accuracy", 0), nvq.get("open_accuracy", 0)]
                 if three else None)

    x = np.arange(len(labels))
    w = 0.22 if three else 0.30
    offsets = ([-w, 0, w] if three else [-w/2, w/2])

    fig, ax = plt.subplots(figsize=(9 if three else 8, 5.5))

    series = [
        (mm_vals,   "Multimodal RAG",     COLORS["mm"]),
        (to_vals,   "Text-only Grounded", COLORS["to"]),
        *( [(open_vals, "Text-only Open", COLORS["open"])] if three else [] ),
    ]
    for idx, (vals, label, color) in enumerate(series):
        b = ax.bar(x + offsets[idx], vals, w, label=label, color=color, alpha=0.88, zorder=3)
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Δ annotation: MM vs best text-only baseline
    for i in range(len(labels)):
        best_to = max(to_vals[i], open_vals[i] if three else 0)
        delta   = mm_vals[i] - best_to
        if abs(delta) > 0.01:
            top_y = max(mm_vals[i], best_to) + 0.07
            ax.annotate(f"Δ+{delta:.0%}", xy=(x[i], top_y),
                        ha="center", fontsize=10, color="#C44E52", fontweight="bold")

    ax.set_ylim(0, 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    title = ("三路对比  —  视觉题 vs 非视觉题" if three else
             "视觉题 vs 非视觉题  —  两种模式准确率对比")
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "fig2_visual_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 – Per-question ANLS heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_per_question_heatmap(data: dict, out_dir: Path):
    details   = data["details"]
    three     = _has_open(data)
    ids       = [r["id"]              for r in details]
    mm_scores = [r["multimodal_anls"] for r in details]
    to_scores = [r["text_only_anls"]  for r in details]
    vis_flags = [r["requires_visual"] for r in details]

    rows      = [mm_scores, to_scores]
    row_labels= ["Multimodal\nRAG", "Text-only\nGrounded"]
    if three:
        open_scores = [r.get("text_open_anls", 0) for r in details]
        rows.append(open_scores)
        row_labels.append("Text-only\nOpen")

    n_rows = len(rows)
    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(max(18, len(ids) * 0.4), 1.4 * n_rows + 0.6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)

    for tick, v in zip(ax.get_xticklabels(), vis_flags):
        if v:
            tick.set_color(COLORS["err"])
            tick.set_fontweight("bold")

    # legend patch to explain the red labels
    red_patch = mpatches.Patch(color=COLORS["err"], label="视觉题 (requires visual)")
    ax.legend(handles=[red_patch], fontsize=8, loc="upper right")

    # value annotations inside cells (skip if too many columns — unreadable)
    if len(ids) <= 50:
        for col in range(len(ids)):
            for row_idx, scores in enumerate(rows):
                s = scores[col]
                txt_color = "black" if 0.3 < s < 0.8 else "white"
                ax.text(col, row_idx, f"{s:.1f}", ha="center", va="center",
                        fontsize=6, color=txt_color)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_label("ANLS", fontsize=9)

    ax.set_title("每道题 ANLS 分数热图  (红色题号 = 视觉题)", fontsize=12, pad=8)
    fig.subplots_adjust(bottom=0.25)
    out = out_dir / "fig3_per_question_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Text-only failure mode pie chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_to_failure_pie(data: dict, out_dir: Path):
    details = data["details"]
    to_failures = [r for r in details if not r["to_correct"]]

    def classify(r):
        ans = r["text_only_answer"]
        if any(kw in ans for kw in ["无法回答", "无法", "不知道", "没有提供", "文档中未", "无相关"]):
            return "拒答（无图信息）"
        # check if gold substring actually present (metric edge case)
        for g in r["gold_answers"]:
            if g.lower() in ans.lower():
                return "指标误判（实际答对）"
        return "答案有误（推理错误）"

    counts = Counter(classify(r) for r in to_failures)
    total_failed = len(to_failures)
    total_all    = len(details)

    labels = list(counts.keys())
    sizes  = list(counts.values())
    pie_colors = [COLORS["to"], COLORS["mid"], COLORS["err"]][:len(labels)]

    fig, (ax_pie, ax_info) = plt.subplots(1, 2, figsize=(10, 4.5),
                                           gridspec_kw={"width_ratios": [1.4, 1]})

    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=pie_colors, startangle=140,
        textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax_pie.set_title(f"Text-only RAG 失败原因分析\n(失败 {total_failed}/{total_all} 题)", fontsize=12)

    # right panel: summary table
    ax_info.axis("off")
    table_data = [["失败原因", "题数", "占失败"]] + \
                 [[k, str(v), f"{v/total_failed:.0%}"] for k, v in counts.items()] + \
                 [["合计失败", str(total_failed), f"{total_failed/total_all:.0%} 总题"]]
    tbl = ax_info.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center", loc="center",
        bbox=[0, 0.1, 1, 0.8],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(fontweight="bold")
    ax_info.set_title("明细", fontsize=11, pad=4)

    plt.tight_layout()
    out = out_dir / "fig4_to_failure_pie.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 – McNemar test summary (visual + non-visual + overall)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_mcnemar_summary(data: dict, out_dir: Path):
    from scipy.stats import binomtest

    details = data["details"]

    def run_mcnemar(subset, label):
        n = len(subset)
        bc = sum(1 for r in subset if     r["mm_correct"] and     r["to_correct"])
        mo = sum(1 for r in subset if     r["mm_correct"] and not r["to_correct"])
        to = sum(1 for r in subset if not r["mm_correct"] and     r["to_correct"])
        bw = sum(1 for r in subset if not r["mm_correct"] and not r["to_correct"])
        # McNemar exact test = binomial test on discordant pairs (mo vs to)
        # H0: P(MM✅TO❌) = 0.5  among discordant pairs
        discordant = mo + to
        use_exact = (n <= 25)
        try:
            if discordant == 0:
                p, stat = 1.0, 0.0
            else:
                res = binomtest(mo, discordant, 0.5, alternative="two-sided")
                p   = res.pvalue
                # chi2-approx statistic for display (with continuity correction)
                stat = (abs(mo - to) - 1) ** 2 / max(discordant, 1) if not use_exact else mo
        except Exception:
            p, stat = float("nan"), float("nan")
        mm_acc = sum(r["mm_correct"] for r in subset) / n
        to_acc = sum(r["to_correct"] for r in subset) / n
        return {
            "label": label, "n": n,
            "mm_acc": mm_acc, "to_acc": to_acc,
            "delta": mm_acc - to_acc,
            "bc": bc, "mo": mo, "to_": to, "bw": bw,
            "stat": stat, "p": p,
            "exact": use_exact,
            "sig": p < 0.05 if not np.isnan(p) else False,
        }

    subsets = [
        (details, "全部 (n=25)"),
        ([r for r in details if r["requires_visual"]],      "视觉题 (n=11)"),
        ([r for r in details if not r["requires_visual"]], "非视觉题 (n=14)"),
    ]
    rows = [run_mcnemar(s, lbl) for s, lbl in subsets]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                              gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: grouped bar showing MM vs TO per subset
    ax = axes[0]
    x = np.arange(len(rows))
    w = 0.30
    b1 = ax.bar(x - w/2, [r["mm_acc"] for r in rows], w,
                label="Multimodal RAG", color=COLORS["mm"], alpha=0.88, zorder=3)
    b2 = ax.bar(x + w/2, [r["to_acc"] for r in rows], w,
                label="Text-only RAG",  color=COLORS["to"], alpha=0.88, zorder=3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f"{h:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i, r in enumerate(rows):
        sig_mark = "* p<0.05" if r["sig"] else "n.s."
        color    = COLORS["ok"] if r["sig"] else "gray"
        top = max(r["mm_acc"], r["to_acc"]) + 0.06
        ax.text(x[i], top, sig_mark, ha="center", va="bottom", fontsize=9,
                color=color, fontweight="bold" if r["sig"] else "normal")
    ax.set_ylim(0, 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("McNemar 检验  —  MM vs TO 各子集对比", fontsize=12, pad=10)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: contingency table summary
    ax2 = axes[1]
    ax2.axis("off")
    col_labels = ["子集", "n", "MM+TO+", "MM+TO-", "MM-TO+", "p值", "显著"]
    cell_data = []
    for r in rows:
        p_str  = f"{r['p']:.4f}" if not np.isnan(r["p"]) else "—"
        sig_str = "[Sig] 是" if r["sig"] else "[-] 否"
        cell_data.append([r["label"], str(r["n"]),
                          str(r["bc"]), str(r["mo"]), str(r["to_"]),
                          p_str, sig_str])
    tbl = ax2.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0.15, 1, 0.75],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#D0D8E8")
            cell.set_text_props(fontweight="bold")
        elif col == 6 and row > 0:
            cell.set_facecolor("#D4EDDA" if "[Sig]" in cell.get_text().get_text() else "#F8D7DA")
    ax2.set_title("配对差异表（McNemar contingency）", fontsize=10, pad=4)

    plt.tight_layout()
    out = out_dir / "fig5_mcnemar_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
    return rows  # return for printing to console


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 – Accuracy by difficulty level (3-way, only if by_difficulty present)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_difficulty(data: dict, out_dir: Path):
    diff_data = data.get("by_difficulty")
    if not diff_data:
        print("  [skip] fig7: no by_difficulty in results.json")
        return

    three = _has_open(data)
    order = ["easy", "medium", "hard"]
    keys  = [k for k in order if k in diff_data] + [k for k in diff_data if k not in order]

    labels   = [f"{k}\n(n={diff_data[k]['count']})" for k in keys]
    mm_acc   = [diff_data[k]["mm_accuracy"]              for k in keys]
    to_acc   = [diff_data[k]["to_accuracy"]              for k in keys]
    open_acc = [diff_data[k].get("open_accuracy", 0)     for k in keys] if three else None

    x = np.arange(len(keys))
    w = 0.22 if three else 0.30
    offsets = ([-w, 0, w] if three else [-w/2, w/2])

    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        (mm_acc,   "Multimodal RAG",     COLORS["mm"]),
        (to_acc,   "Text-only Grounded", COLORS["to"]),
        *( [(open_acc, "Text-only Open", COLORS["open"])] if three else [] ),
    ]
    for idx, (vals, label, color) in enumerate(series):
        b = ax.bar(x + offsets[idx], vals, w, label=label, color=color, alpha=0.88, zorder=3)
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("按难度分层  —  三路模式对比" if three else "按难度分层  —  准确率对比",
                 fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "fig7_difficulty.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Loading results from: {RESULTS_PATH}")
    if not RESULTS_PATH.exists():
        print("ERROR: results.json not found. Run evaluation first.")
        sys.exit(1)

    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  {data['total_questions']} questions, model={data['model']}")
    three = _has_open(data)
    print(f"  3-way evaluation: {'yes (MM + TO-Grounded + TO-Open)' if three else 'no (MM + TO only)'}\n")
    print(f"Output directory: {OUT_DIR}\n")

    print("Generating fig1 – main comparison bar chart ...")
    fig1_main_comparison(data, OUT_DIR)

    print("Generating fig2 – visual/non-visual split ...")
    fig2_visual_split(data, OUT_DIR)

    print("Generating fig3 – per-question ANLS heatmap ...")
    fig3_per_question_heatmap(data, OUT_DIR)

    print("Generating fig4 – text-only failure pie chart ...")
    fig4_to_failure_pie(data, OUT_DIR)

    print("Generating fig5 – McNemar test summary ...")
    mcnemar_rows = fig5_mcnemar_summary(data, OUT_DIR)

    print("Generating fig7 – accuracy by difficulty ...")
    fig7_difficulty(data, OUT_DIR)

    # ── Console summary ────────────────────────────────────────────────────────
    ov = data["overall"]
    print("\n" + "=" * 65)
    print("Overall Accuracy Summary")
    print("=" * 65)
    print(f"  Multimodal RAG    : {ov['multimodal_accuracy']:.0%}")
    print(f"  Text-only Grounded: {ov['text_only_accuracy']:.0%}")
    if three:
        print(f"  Text-only Open    : {ov.get('text_open_accuracy', 0):.0%}")

    print("\n" + "=" * 65)
    print("McNemar Test Results (MM vs TO-Grounded)")
    print("=" * 65)
    for r in mcnemar_rows:
        sig = "[Sig] p<0.05" if r["sig"] else "[n.s.]"
        print(f"  {r['label']:22s}  MM={r['mm_acc']:.0%}  TO={r['to_acc']:.0%}"
              f"  Δ=+{r['delta']:.0%}  p={r['p']:.4f}  {sig}")

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Files:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    # allow running as script directly from project root
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
