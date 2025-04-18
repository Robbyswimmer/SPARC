import re
from typing import List

def normalize_text(s: str) -> List[str]:
    """
    Lowercase, remove punctuation (except alphanumerics and whitespace), then split.
    Returns list of tokens.
    """
    s = s.lower()
    # Remove punctuation
    s = re.sub(r'[^a-z0-9\s]', '', s)
    tokens = s.split()
    return tokens

def compute_em(gold: str, pred: str) -> float:
    """
    Exact Match: 1.0 if gold == pred after normalization, else 0.0
    """
    return float(normalize_text(gold) == normalize_text(pred))

def compute_f1(gold: str, pred: str) -> float:
    """
    Word-level F1 score between gold and pred strings.
    """
    gold_tokens = normalize_text(gold)
    pred_tokens = normalize_text(pred)
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = set(gold_tokens) & set(pred_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_qa_score(gold: str, pred: str) -> float:
    """
    Combine EM and F1: weighted sum (e.g., 0.5*EM + 0.5*F1).
    """
    em = compute_em(gold, pred)
    f1 = compute_f1(gold, pred)
    return 0.5 * em + 0.5 * f1