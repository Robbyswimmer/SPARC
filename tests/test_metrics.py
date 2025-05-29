# tests/test_metrics.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.metrics import compute_exact_match, compute_f1, compute_qa_score_multi

def test_em_case_insensitive():
    pred = "Paris"
    refs = ["paris", "PARIS"]
    em = compute_exact_match(pred, refs[0])
    print(em)
    assert compute_exact_match(pred, refs[0]) == 1

def test_f1_partial():
    print(compute_f1("capital of france", "The capital is France"))
    assert compute_f1("capital of france", "The capital is France") > 0

def test_multi_reference():
    refs = ["Blue whale", "The blue whale"]
    pred = "A blue whale"
    score = compute_qa_score_multi(refs, pred)
    print(score)
    assert 0 <= score <= 1 and score > 0.5
