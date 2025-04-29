#!/usr/bin/env python3
# requires: conda activate streamqa
"""
Tests for data registry and dataset adapters.

Verifies that each dataset adapter in the registry:
1. Follows the standardized output format
2. Emits sufficient number of examples
3. Properly handles tokenization and chunking
"""

import pytest
import os
import sys
from typing import Dict, Any
import tqdm
import time

# Add the src directory to the path so we can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Import the module to test
from src.data.data_registry import (
    DATASET_ADAPTERS, 
    get_dataset_iterator,
    narrativeqa_adapter,
    hotpotqa_adapter,
    triviaqa_adapter,
    nq_adapter,
)

# Import tokenizer used by the environment
from transformers import AutoTokenizer

# Define constants for tests
MIN_EXAMPLES = 10000  # Minimum number of examples each dataset should provide
SAMPLE_SIZE = 100     # Number of examples to check for format validation
TIMEOUT = 300         # Maximum time (seconds) to wait for dataset loading

# Define fixture for tokenizer
@pytest.fixture
def tokenizer():
    """Fixture providing the tokenizer used by the environment."""
    return AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")

# Helper function to validate example format
def validate_example_format(example: Dict[str, Any]) -> bool:
    """Validate that an example follows the expected format.
    
    Args:
        example: Dictionary containing the example data
        
    Returns:
        True if the example is valid, False otherwise
    """
    # Check required keys
    required_keys = ["doc_chunks", "question_ids", "answers", "meta"]
    if not all(key in example for key in required_keys):
        return False
    
    # Validate doc_chunks format
    if not isinstance(example["doc_chunks"], list) or not all(isinstance(chunk, list) for chunk in example["doc_chunks"]):
        return False
    
    # Validate question_ids format
    if not isinstance(example["question_ids"], list) or not all(isinstance(token_id, int) for token_id in example["question_ids"]):
        return False
    
    # Validate answers format
    if not isinstance(example["answers"], list) or not all(isinstance(answer, str) for answer in example["answers"]):
        return False
    
    # Validate meta format
    if not isinstance(example["meta"], dict) or "dataset" not in example["meta"]:
        return False
    
    return True

# Test that all registered adapters exist and are callable
def test_adapter_registry_completeness():
    """Test that all dataset adapters are properly registered."""
    # Expected datasets
    expected_datasets = ["narrativeqa", "hotpotqa", "triviaqa", "nq_long"]
    
    # Check all expected datasets are in registry
    for dataset in expected_datasets:
        assert dataset in DATASET_ADAPTERS, f"Dataset {dataset} not found in registry"
    
    # Check that all registered adapters are callable
    for adapter_name, adapter_fn in DATASET_ADAPTERS.items():
        assert callable(adapter_fn), f"Adapter for {adapter_name} is not callable"

# Parameterized test for each dataset adapter
@pytest.mark.slow  # Mark as slow test
@pytest.mark.parametrize(
    "dataset_name,adapter_fn", [
        ("narrativeqa", narrativeqa_adapter),
        ("hotpotqa", hotpotqa_adapter),
        ("triviaqa", triviaqa_adapter),
        ("nq_long", nq_adapter),
    ]
)
def test_dataset_adapter_output_format(dataset_name: str, adapter_fn, tokenizer):
    """Test that each dataset adapter outputs the expected format."""
    # Get iterator from the adapter
    iterator = adapter_fn(tokenizer, split="train", streaming=True, chunk_size=256)
    
    # Validate a sample of examples
    count = 0
    for example in iterator:
        assert validate_example_format(example), f"Invalid example format from {dataset_name}"
        count += 1
        if count >= SAMPLE_SIZE:
            break
    
    # Make sure we got at least some examples
    assert count > 0, f"No examples generated from {dataset_name}"

# Test that each dataset provides enough examples
@pytest.mark.slow  # Mark as slow test
@pytest.mark.timeout(TIMEOUT)  # Apply timeout to prevent tests from hanging
@pytest.mark.parametrize(
    "dataset_name", [
        "narrativeqa",
        "hotpotqa",
        "triviaqa",
        "nq_long",
    ]
)
def test_dataset_adapter_min_examples(dataset_name: str, tokenizer):
    """Test that each dataset adapter provides at least MIN_EXAMPLES examples."""
    # Get iterator using the registry
    iterator = get_dataset_iterator(dataset_name, tokenizer, split="train", chunk_size=256)
    
    # Count examples with progress bar
    print(f"\nCounting examples for {dataset_name}...")
    count = 0
    start_time = time.time()
    
    for _ in tqdm.tqdm(iterator, total=MIN_EXAMPLES, desc=dataset_name, unit="examples"):
        count += 1
        if count >= MIN_EXAMPLES:
            break
    
    end_time = time.time()
    duration = end_time - start_time
    examples_per_second = count / duration if duration > 0 else 0
    
    print(f"\n{dataset_name} generated {count} examples in {duration:.2f}s ({examples_per_second:.2f} examples/s)")
    
    # Assert that we have the minimum number of examples
    assert count >= MIN_EXAMPLES, f"{dataset_name} only provided {count} examples, expected at least {MIN_EXAMPLES}"

# Test mixed stream with all datasets
@pytest.mark.slow  # Mark as slow test
def test_mixed_stream_with_all_datasets(tokenizer):
    """Test that the mixed stream combines examples from all datasets."""
    from src.data.data_registry import mixed_stream
    
    # Get all dataset names
    all_datasets = list(DATASET_ADAPTERS.keys())
    
    # Create mixed stream
    stream_iter = mixed_stream(all_datasets, tokenizer, split="train", chunk_size=256, seed=42)
    
    # Count examples by dataset
    dataset_counts = {dataset: 0 for dataset in all_datasets}
    total_count = 0
    
    for _ in range(1000):  # Check 1000 examples
        example = next(stream_iter)
        dataset = example["meta"]["dataset"]
        dataset_counts[dataset] += 1
        total_count += 1
    
    # Print distribution
    print("\nMixed stream dataset distribution:")
    for dataset, count in dataset_counts.items():
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        print(f"{dataset}: {count} examples ({percentage:.1f}%)")
    
    # Ensure we have examples from all datasets
    for dataset in all_datasets:
        assert dataset_counts[dataset] > 0, f"No examples from {dataset} in mixed stream"

if __name__ == "__main__":
    # Allow running specific tests from command line
    pytest.main([__file__, "-v"])
