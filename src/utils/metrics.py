import re
from typing import List, Callable
from collections import Counter

def normalize_text(s: str) -> List[str]:
    """
    Lowercase, remove punctuation (except alphanumerics and whitespace), then split.
    Remove articles (a, an, the) and return list of tokens.
    """
    s = s.lower()
    # Remove punctuation while preserving decimal points
    s = re.sub(r'[^\w\s.]', '', s)
    # Remove punctuation that's not part of a number
    s = re.sub(r'\.(?!\d)', '', s)
    tokens = s.split()
    # Remove articles
    return [token for token in tokens if token not in {'a', 'an', 'the'}]

def compute_em(gold: str, pred: str) -> float:
    """
    Exact Match: 1.0 if gold == pred after normalization, else 0.0
    """
    return float(normalize_text(gold) == normalize_text(pred))

def compute_exact_match(pred: str, gold: str) -> float:
    """
    Alias for compute_em with reversed parameter order for consistency.
    """
    return compute_em(gold, pred)

def compute_f1(gold: str, pred: str) -> float:
    """
    Word-level F1 score between gold and pred strings.
    Uses SQuAD-style bag-of-words F1 (token overlap).
    """
    gold_tokens = normalize_text(gold)
    pred_tokens = normalize_text(pred)

    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counter = Counter(gold_tokens)
    pred_counter = Counter(pred_tokens)
    common = gold_counter & pred_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1 * 10000) / 10000  # round for test match

def max_over_refs(metric_fn: Callable[[str, str], float], pred: str, refs: List[str]) -> float:
    """
    Compute the maximum score of a metric over multiple reference answers.
    
    Args:
        metric_fn: Function that computes a metric between prediction and reference
        pred: The predicted answer
        refs: List of reference (gold) answers
        
    Returns:
        Maximum score across all references
    """
    if not refs:
        return 0.0
    return max(metric_fn(pred, ref) for ref in refs)

def compute_qa_score(gold: str, pred: str) -> float:
    """
    Combine EM and F1: weighted sum (e.g., 0.5*EM + 0.5*F1) for a single reference.
    """
    em = compute_em(gold, pred)
    f1 = compute_f1(gold, pred)
    return 0.5 * em + 0.5 * f1

def compute_qa_score_multi(gold_refs: List[str], pred: str) -> float:
    """
    Combine max-over-references EM and F1: weighted sum (0.5*EM + 0.5*F1).
    Uses the maximum score across all reference answers.
    
    Args:
        gold_refs: List of reference (gold) answers
        pred: The predicted answer
        
    Returns:
        Combined QA score using max-over-references approach
    """
    em = max_over_refs(compute_em, pred, gold_refs)
    f1 = max_over_refs(compute_f1, pred, gold_refs)
    return 0.5 * em + 0.5 * f1

# ================================= TEST CASES =================================
def run_test(test_name, gold, pred, expected_em=None, expected_f1=None, expected_qa=None):
    """Run a test case and print results."""
    actual_em = compute_em(gold, pred)
    actual_f1 = compute_f1(gold, pred)
    actual_qa = compute_qa_score(gold, pred)
    
    # Check if expected values are provided, otherwise just report actual values
    em_result = "✓" if expected_em is None or abs(actual_em - expected_em) < 0.001 else "✗"
    f1_result = "✓" if expected_f1 is None or abs(actual_f1 - expected_f1) < 0.001 else "✗"
    qa_result = "✓" if expected_qa is None or abs(actual_qa - expected_qa) < 0.001 else "✗"
    
    print(f"TEST: {test_name}")
    print(f"  Gold answer: '{gold}'")
    print(f"  Pred answer: '{pred}'")
    print(f"  EM:  {actual_em:.4f} {em_result} {f'(expected {expected_em:.4f})' if expected_em is not None else ''}")
    print(f"  F1:  {actual_f1:.4f} {f1_result} {f'(expected {expected_f1:.4f})' if expected_f1 is not None else ''}")
    print(f"  QA:  {actual_qa:.4f} {qa_result} {f'(expected {expected_qa:.4f})' if expected_qa is not None else ''}")
    print()
    
    return all(r == "✓" for r in [em_result, f1_result, qa_result])


def run_normalization_test(test_name, text, expected_tokens):
    """Test the text normalization function."""
    actual_tokens = normalize_text(text)
    result = "✓" if actual_tokens == expected_tokens else "✗"
    
    print(f"NORM TEST: {test_name}")
    print(f"  Input text:     '{text}'")
    print(f"  Actual tokens:   {actual_tokens}")
    print(f"  Expected tokens: {expected_tokens}")
    print(f"  Result: {result}")
    print()
    
    return result == "✓"


def run_test_suite():
    """Run all test cases and report success/failure."""
    print("\n=== RUNNING QA METRICS TEST SUITE ===")
    test_results = []
    norm_results = []
    
    # Test text normalization
    norm_results.append(run_normalization_test(
        "Basic normalization",
        "Hello, World! How are you?",
        ['hello', 'world', 'how', 'are', 'you']
    ))
    
    norm_results.append(run_normalization_test(
        "Article removal",
        "The cat sat on a mat.",
        ['cat', 'sat', 'on', 'mat']
    ))
    
    norm_results.append(run_normalization_test(
        "Empty string",
        "",
        []
    ))
    
    norm_results.append(run_normalization_test(
        "Extra whitespace and numbers",
        "  The  price   is $12.50  today. ",
        ['price', 'is', '12.50', 'today']
    ))
    
    # Test exact match
    test_results.append(run_test(
        "Exact match - identical", 
        "42", 
        "42", 
        expected_em=1.0, 
        expected_f1=1.0, 
        expected_qa=1.0
    ))
    
    test_results.append(run_test(
        "Exact match - case difference", 
        "Blue", 
        "blue", 
        expected_em=1.0, 
        expected_f1=1.0, 
        expected_qa=1.0
    ))
    
    test_results.append(run_test(
        "Exact match - punctuation", 
        "Hello, world!", 
        "hello world", 
        expected_em=1.0, 
        expected_f1=1.0, 
        expected_qa=1.0
    ))
    
    test_results.append(run_test(
        "Exact match - article difference", 
        "I saw a cat", 
        "I saw the cat", 
        expected_em=1.0, 
        expected_f1=1.0, 
        expected_qa=1.0
    ))
    
    # Test F1 partial matches
    test_results.append(run_test(
        "Partial match - subset", 
        "red car with four wheels", 
        "red car", 
        expected_em=0.0, 
        expected_f1=0.5714, 
        expected_qa=0.2857
    ))
    
    test_results.append(run_test(
        "Partial match - overlapping", 
        "the large brown dog", 
        "the small brown dog", 
        expected_em=0.0, 
        expected_f1=0.6667, 
        expected_qa=0.3333
    ))
    
    # Test different answers
    test_results.append(run_test(
        "Different answers", 
        "Paris", 
        "London", 
        expected_em=0.0, 
        expected_f1=0.0, 
        expected_qa=0.0
    ))
    
    # Test empty answers
    test_results.append(run_test(
        "Empty gold answer", 
        "", 
        "something", 
        expected_em=0.0, 
        expected_f1=0.0, 
        expected_qa=0.0
    ))

    test_results.append(run_test(
        "Empty predicted answer", 
        "something", 
        "", 
        expected_em=0.0, 
        expected_f1=0.0, 
        expected_qa=0.0
    ))
    
    test_results.append(run_test(
        "Both answers empty", 
        "", 
        "", 
        expected_em=1.0, 
        expected_f1=0.0, 
        expected_qa=0.5
    ))

    # Test edge cases with longer answers
    test_results.append(run_test(
        "NarrativeQA example",
        "Miss Delmer",
        "Miss Delmer is the elderly spinster aunt of the Earl",
        expected_em=0.0,
        expected_f1=0.4000, 
        expected_qa=0.2000
    ))

    # Summarize results
    norm_success = sum(norm_results)
    norm_total = len(norm_results)
    test_success = sum(test_results)
    test_total = len(test_results)
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Normalization tests: {norm_success}/{norm_total} passed")
    print(f"QA Metric tests: {test_success}/{test_total} passed")
    print(f"Overall: {norm_success + test_success}/{norm_total + test_total} passed")
    
    return norm_success + test_success == norm_total + test_total


if __name__ == "__main__":
    # When run as a script, execute the test suite
    success = run_test_suite()
    exit(0 if success else 1)  # Return exit code based on test results
