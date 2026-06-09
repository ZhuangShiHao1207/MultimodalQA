"""
evaluation/datasets/annotate_dataset.py

Adds two fields to every QA item in self_built_qa.json:
  - difficulty: "easy" | "medium" | "hard"
  - expected_elements: { images: [], text_keywords: [], tables: [] }

Difficulty rubric:
  easy   - single source, direct lookup, no inference required
  medium - requires understanding trend/comparison between 2 items,
           reading a chart label, or combining 2 pieces of info
  hard   - cross-document or cross-figure comparison, calculation,
           requires reasoning across multiple elements

Run:
  python3 evaluation/datasets/annotate_dataset.py
  (writes to self_built_qa.json in-place after making a .bak)
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
QA_PATH = HERE / "self_built_qa.json"

# ── Manual annotations ────────────────────────────────────────────────────────
# Format: { id: (difficulty, keywords) }
# keywords: for text/table questions, key terms that MUST appear in the
#           retrieved chunk to count as a correct retrieval.
# Figure questions get images filled in later (need actual filenames from
# the backend after indexing).
ANNOTATIONS = {
    # ── 测试数据（K-Means/GMM）────────────────────────────────────────────────
    "q01": ("easy",   ["目标函数", "K-Means", "簇内"]),
    "q02": ("easy",   ["46", "K-Means++", "迭代"]),
    "q03": ("easy",   ["0.5937", "Random", "测试"]),
    "q04": ("medium", []),   # figure
    "q05": ("easy",   ["Full", "训练", "准确率"]),
    "q06": ("easy",   ["0.5376", "Spherical", "测试"]),
    "q07": ("easy",   ["0.6350", "Full", "Random"]),
    "q08": ("medium", ["Random", "K-Means++", "Ground Truth", "ACC"]),
    "q09": ("easy",   ["GMM", "K-Means", "准确率"]),
    "q10": ("easy",   ["126.15", "Full", "训练时间"]),
    "q11": ("medium", []),   # figure - ordering from curve
    "q12": ("easy",   ["E步", "后验概率", "责任度"]),
    "q26": ("medium", ["迭代", "K-Means++", "Random"]),
    "q27": ("medium", ["M步", "混合系数", "软分配"]),
    "q28": ("medium", []),   # figure
    "q29": ("easy",   ["32.47", "Diagonal", "训练时间"]),
    "q30": ("easy",   ["MNIST", "数据集"]),
    "q89": ("easy",   ["0.5937", "K-Means++", "训练"]),
    "q90": ("easy",   ["0.5516", "Diagonal", "测试"]),
    "q91": ("easy",   ["0.5763", "Full", "K-Means++"]),
    "q96": ("medium", []),   # figure
    "q97": ("hard",   []),   # figure - early-iteration comparison
    "q98": ("medium", []),   # figure
    "q99": ("medium", []),   # figure - read approximate iteration count
    "q100":("hard",   []),   # figure - compare two convergence levels

    # ── 大数据报告（IMDB）────────────────────────────────────────────────────
    "q13": ("medium", []),   # figure - read bar chart
    "q14": ("medium", []),   # figure - compare two bars
    "q15": ("medium", []),   # figure - read min of val loss
    "q16": ("medium", []),   # figure - read initial value
    "q17": ("easy",   []),   # figure - read confusion matrix cell
    "q18": ("hard",   []),   # figure - compute recall from matrix
    "q19": ("easy",   []),   # figure - read confusion matrix cell
    "q20": ("hard",   []),   # figure - compare FP across 4 matrices
    "q21": ("easy",   ["TF-IDF", "特征提取"]),
    "q22": ("easy",   ["BiLSTM", "双向"]),
    "q23": ("easy",   ["IMDB", "25000"]),
    "q24": ("medium", []),   # figure
    "q25": ("medium", ["传统方法", "Accuracy", "TF-IDF"]),
    "q31": ("easy",   ["0.9526", "ROC_AUC", "LogisticRegression"]),
    "q32": ("easy",   ["0.8647", "Precision", "LinearSVM"]),
    "q33": ("medium", ["RNN-Strong", "RNN", "Accuracy"]),
    "q34": ("medium", []),   # figure
    "q35": ("easy",   ["256", "隐层", "BiLSTM"]),
    "q92": ("easy",   ["0.8849", "Recall", "LogisticRegression"]),
    "q93": ("medium", []),   # figure
    "q94": ("easy",   ["0.8374", "F1", "RNN-Strong"]),
    "q95": ("medium", []),   # figure
    "q101":("easy",   []),   # figure - read TN cell
    "q102":("easy",   []),   # figure - read FN cell
    "q103":("medium", []),   # figure - read approximate bar height
    "q104":("easy",   []),   # figure - read FP cell
    "q105":("medium", []),   # figure

    # ── Infini-attention (2404.07143v2.pdf) ───────────────────────────────────
    "q36": ("easy",   ["114x", "压缩", "Memorizing"]),
    "q37": ("easy",   ["9.65", "PG19", "perplexity"]),
    "q38": ("easy",   ["40.0", "ROUGE-1", "BookSum"]),
    "q39": ("easy",   ["18.5", "Overall", "BookSum"]),
    "q40": ("easy",   ["mixer heads", "gating score", "0.5"]),
    "q41": ("medium", []),   # figure
    "q42": ("medium", []),   # figure - trend direction
    "q43": ("medium", []),   # figure - architectural comparison
    "q44": ("easy",   ["100", "1M", "passkey", "fine-tun"]),
    "q45": ("easy",   ["linear attention", "compressive memory"]),
    "q83": ("easy",   ["11.88", "Transformer-XL", "PG19"]),
    "q84": ("easy",   ["73x", "RMT", "compression"]),
    "q85": ("medium", []),   # figure
    "q86": ("easy",   ["14", "13", "98", "32K", "zero-shot"]),
    "q87": ("easy",   ["15.6", "PRIMERA", "ROUGE-L"]),
    "q88": ("easy",   ["16.2", "BART", "Overall"]),
    "q106":("medium", []),   # figure - read approximate y-value
    "q107":("easy",   []),   # figure - count segments
    "q108":("medium", []),   # figure - distribution description
    "q109":("medium", []),   # figure - read architecture detail

    # ── IPCC AR6 ─────────────────────────────────────────────────────────────
    "q46": ("easy",   ["1.1", "2011-2020", "1850-1900"]),
    "q47": ("easy",   ["59", "GtCO2", "2019", "A.1.4"]),
    "q48": ("medium", []),   # figure - colour trend
    "q49": ("hard",   []),   # figure - read specific legend entry
    "q50": ("easy",   ["C1", "1.5°C", "Table"]),
    "q51": ("hard",   []),   # figure - regional detail at 4°C
    "q52": ("easy",   ["43", "2030", "GHG", "1.5°C"]),
    "q53": ("medium", []),   # figure - read label on implemented policies path
    "q54": ("easy",   ["0.45°C", "1000 GtCO2", "B.5.2"]),
    "q55": ("easy",   ["35%", "9 tCO2", "A.1.5"]),
    "q56": ("medium", []),   # figure - identify panel
    "q57": ("easy",   ["0.15", "0.23", "SSP1-1.9", "sea level"]),
    "q58": ("easy",   ["3.2°C", "A.4.4", "2100"]),
    "q59": ("medium", []),   # figure - trend across warming levels
    "q60": ("easy",   ["1.8", "GtCO2", "A.4.1", "2016"]),
    "q76": ("easy",   ["0.20", "sea level", "1901", "2018"]),
    "q77": ("medium", []),   # figure
    "q78": ("easy",   ["64", "2050", "2°C"]),
    "q79": ("medium", ["four fifths", "1850", "2019", "carbon budget"]),
    "q80": ("medium", []),   # figure
    "q81": ("easy",   ["public", "adaptation", "finance"]),
    "q82": ("easy",   ["0.63", "1.01", "SSP5-8.5", "2100"]),
    "q110":("medium", []),   # figure - colour intensity trend
    "q111":("medium", []),   # figure - colour of drying regions
    "q112":("medium", []),   # figure - read net-zero year
    "q113":("medium", []),   # figure - above/below zero
    "q114":("easy",   ["35", "2035", "2°C"]),
    "q115":("medium", []),   # figure
    "q116":("easy",   []),   # figure - overall colour trend

    # ── IMF 经济博客 ──────────────────────────────────────────────────────────
    "q61": ("medium", []),   # figure - read tallest negative bar
    "q62": ("easy",   ["3.4%", "2026", "增长率"]),
    "q63": ("medium", []),   # figure - read highest line at 2026-03
    "q64": ("easy",   ["3.1%", "参考预测", "2026"]),
    "q65": ("easy",   ["6%", "严峻情景", "通胀"]),
    "q66": ("medium", []),   # figure
    "q67": ("medium", []),   # figure
    "q68": ("medium", []),   # figure - trend direction
    "q69": ("easy",   ["大宗商品", "供给冲击", "能源"]),
    "q69b":("easy",   ["19%", "能源大宗商品", "2026"]),
    "q70": ("medium", []),   # figure - identify smallest negative bar
    "q71": ("easy",   ["衰退", "通胀", "收紧货币"]),
    "q72": ("medium", []),   # figure - read approximate % value
    "q73": ("easy",   ["2.5%", "不利情景"]),
    "q74": ("hard",   []),   # figure - compare two specific bars
    "q75": ("medium", ["转移支付", "财政", "针对性"]),
    "q117":("hard",   []),   # figure - identify flattest line
    "q118":("medium", []),   # figure - read approximate index value
    "q119":("medium", []),   # figure - identify positive bar
    "q120":("medium", []),   # figure - identify inflection point
    "q121":("easy",   []),   # figure - read approximate bar height
    "q122":("medium", []),   # figure - comparative magnitude
    "q123":("medium", []),   # figure - identify monotonically rising line
}

# ── Apply annotations ─────────────────────────────────────────────────────────

def annotate():
    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    missing = []
    for q in data:
        qid = q["id"]
        if qid not in ANNOTATIONS:
            missing.append(qid)
            diff, kws = "easy", []
        else:
            diff, kws = ANNOTATIONS[qid]

        q["difficulty"] = diff
        q.setdefault("expected_elements", {})
        q["expected_elements"].setdefault("images", [])
        q["expected_elements"].setdefault("tables", [])
        # Only set text_keywords for non-figure questions (figure ones will
        # have keywords = [] since they rely on image retrieval)
        q["expected_elements"]["text_keywords"] = kws

    if missing:
        print(f"WARNING: No annotation for: {missing}")

    # Backup then write
    shutil.copy(QA_PATH, QA_PATH.with_suffix(".json.bak"))
    with open(QA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Stats
    total = len(data)
    by_diff = {}
    for q in data:
        d = q.get("difficulty", "?")
        by_diff[d] = by_diff.get(d, 0) + 1
    vis = sum(1 for q in data if q.get("requires_visual"))

    print(f"Annotated {total} questions")
    print(f"  easy={by_diff.get('easy',0)}  medium={by_diff.get('medium',0)}  hard={by_diff.get('hard',0)}")
    print(f"  visual={vis}  non-visual={total-vis}")
    print(f"Saved to {QA_PATH}  (backup: {QA_PATH.with_suffix('.json.bak')})")

if __name__ == "__main__":
    annotate()
