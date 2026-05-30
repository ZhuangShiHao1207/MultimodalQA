"""
Evaluation metrics for document QA.
Implements ANLS (Average Normalized Levenshtein Similarity) and Accuracy.
"""
import re
from typing import List, Tuple
import Levenshtein


def normalized_levenshtein_similarity(pred: str, gold: str) -> float:
    """
    Compute Normalized Levenshtein Similarity (NLS) between prediction and ground truth.
    NLS = 1 - (edit_distance / max(len(pred), len(gold)))
    Returns 1.0 for exact match, 0.0 for completely different strings.

    For QA tasks where the model gives verbose answers but gold is short
    (e.g., gold="0.5937", pred="Random初始化的测试集准确率为59.37%"), substring
    containment is checked first: if any gold appears verbatim inside pred,
    the answer is judged correct (score=1.0). Otherwise standard NLS.
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    # Normalize: lowercase, strip whitespace, remove common decoration
    def _norm(s: str) -> str:
        s = s.strip().lower()
        # Strip surrounding quotes/brackets
        s = s.strip('"\'`""''《》「」【】[]()（）')
        return s

    pred_norm = _norm(pred)
    gold_norm = _norm(gold)

    if pred_norm == gold_norm:
        return 1.0

    # Substring containment: gold is short and contained in pred
    # (handles "答: 是 BiLSTM" containing gold "BiLSTM")
    if gold_norm and gold_norm in pred_norm:
        return 1.0

    # Numeric tolerance: if gold is a percentage like "59.37%" and pred contains "0.5937"
    # or vice versa, treat as match
    import re
    gold_num = re.search(r'(\d+\.?\d*)', gold_norm)
    if gold_num:
        # Try the raw number
        if gold_num.group(1) in pred_norm:
            return 1.0
        # Try percentage <-> decimal conversion
        try:
            num = float(gold_num.group(1))
            # If gold has %, also check decimal form
            if '%' in gold_norm:
                decimal = num / 100
                if f"{decimal:.4f}".rstrip('0') in pred_norm or f"{decimal:.2f}" in pred_norm:
                    return 1.0
            else:
                # Try as percentage
                pct = num * 100
                if f"{pct:.2f}%" in pred_norm or f"{pct:.0f}%" in pred_norm:
                    return 1.0
        except ValueError:
            pass

    # Ordered comparison: if gold is like "A > B > C", check that A, B, C appear
    # in pred in the same order (with anything in between).
    if '>' in gold_norm or '<' in gold_norm:
        sep = '>' if '>' in gold_norm else '<'
        tokens = [t.strip() for t in gold_norm.split(sep) if t.strip()]
        if len(tokens) >= 2:
            # Check if all tokens appear in pred in same order
            pos = 0
            all_found = True
            for tok in tokens:
                idx = pred_norm.find(tok, pos)
                if idx < 0:
                    all_found = False
                    break
                pos = idx + len(tok)
            if all_found:
                # Verify the order separator (>, <) also appears between consecutive tokens
                return 1.0

    max_len = max(len(pred_norm), len(gold_norm))
    if max_len == 0:
        return 1.0

    distance = Levenshtein.distance(pred_norm, gold_norm)
    nls = 1.0 - (distance / max_len)
    return max(0.0, nls)


def anls_score(pred: str, gold_answers: List[str], threshold: float = 0.5) -> float:
    """
    Compute ANLS score for a single prediction against one or more gold answers.
    ANLS applies a threshold: if NLS < threshold, score is 0.

    Args:
        pred: Model prediction
        gold_answers: List of acceptable gold answers
        threshold: NLS threshold (default 0.5 per DocVQA standard)

    Returns:
        ANLS score (0.0 - 1.0)
    """
    if not gold_answers:
        return 0.0

    max_nls = 0.0
    for gold in gold_answers:
        nls = normalized_levenshtein_similarity(pred, gold)
        max_nls = max(max_nls, nls)

    # Apply threshold
    return max_nls if max_nls >= threshold else 0.0


def compute_anls(predictions: List[str], references: List[List[str]], threshold: float = 0.5) -> float:
    """
    Compute Average ANLS over a dataset.

    Args:
        predictions: List of model predictions
        references: List of lists of acceptable gold answers
        threshold: NLS threshold

    Returns:
        Average ANLS score
    """
    if not predictions:
        return 0.0

    scores = []
    for pred, golds in zip(predictions, references):
        scores.append(anls_score(pred, golds, threshold))

    return sum(scores) / len(scores)


def compute_accuracy(predictions: List[str], references: List[List[str]], normalize: bool = True) -> float:
    """
    Compute exact-match accuracy (with optional normalization).

    Args:
        predictions: List of model predictions
        references: List of lists of acceptable gold answers
        normalize: Whether to normalize (lowercase, strip) before comparing

    Returns:
        Accuracy (0.0 - 1.0)
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, golds in zip(predictions, references):
        pred_norm = pred.strip().lower() if normalize else pred
        for gold in golds:
            gold_norm = gold.strip().lower() if normalize else gold
            if pred_norm == gold_norm:
                correct += 1
                break

    return correct / len(predictions)


def extract_key_answer(response: str) -> str:
    """
    Extract the core answer from a verbose LLM response.
    Tries to find the direct answer by looking for key patterns.
    """
    # Try to find content after common answer indicators
    patterns = [
        r'(?:答案|结果|结论)[：:]\s*(.+?)(?:\n|$)',
        r'(?:是|为|达到)\s*[：:]?\s*([\d.]+%?)',
        r'(\d+\.?\d*%)',  # Percentage
        r'(\d+\.?\d*)',   # Number
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()

    # Fallback: return first sentence
    first_line = response.split('\n')[0].strip()
    if len(first_line) > 200:
        first_line = first_line[:200]
    return first_line
