# scripts/verify_upper_bound.py

import os
import sys
import time
import re
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from src.utils.llm_interface import LLMInterface
from src.utils.model_paths import llama_31_8b
from src.utils.metrics import compute_exact_match, compute_f1, compute_qa_score

# Constants
MAX_CONTEXT_TOKENS = 2048  # Maximum context window size
MAX_GENERATION_TOKENS = 256  # Maximum tokens to generate for answers

# --- Configuration ---
MODEL_PATH = llama_31_8b
TOKENIZER_NAME = "NousResearch/Meta-Llama-3.1-8B"
MAX_CTX = 2048
MAX_TOKENS_PER_DOC = 1700  # Leave room for question and generation
MAX_GENERATION_TOKENS = 100

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True, add_bos_token=True, add_eos_token=True)
llm = LLMInterface(model_path=MODEL_PATH, n_ctx=MAX_CTX, n_threads=4, verbose=False)

def verify_upper_bound(doc_text: str, question: str, gold_answer: str) -> None:
    """
    Query the LLM with the full document + question, then compute EM and F1.
    """
    # Tokenize document to count tokens
    doc_tokens = tokenizer.encode(doc_text)

    # precise truncation logic
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    space_for_doc = MAX_CTX - len(question_ids) - MAX_GENERATION_TOKENS
    doc_tokens = doc_tokens[:space_for_doc]

    # Truncate if needed
    if len(doc_tokens) > MAX_TOKENS_PER_DOC:
        print(f"Document too long ({len(doc_tokens)} tokens). Truncating to {MAX_TOKENS_PER_DOC} tokens.")
        doc_tokens = doc_tokens[:MAX_TOKENS_PER_DOC]
        doc_text = tokenizer.decode(doc_tokens, skip_special_tokens=True)
        print(f"Truncated document length: {len(doc_tokens)} tokens")
    
    # Create system prompt and user prompt
    system_prompt = "You are a helpful, accurate, and concise assistant. Answer questions based on the provided document."
    user_prompt = f"Document: {doc_text}\n\nQuestion: {question}\n\nProvide a direct and concise answer."
    
    # Time the generation
    start_time = time.time()
    
    # Generate answer using the improved interface
    predicted_answer = llm.generate_text(
        user_message=user_prompt,
        system_prompt=system_prompt,
        max_tokens=MAX_GENERATION_TOKENS,
        temperature=0.0,
        stop=["\n\n", "Human:", "Document:", "Question:", "<s>"]
    )
    
    # Clean up the answer
    predicted_answer = predicted_answer.strip()
    # Remove any prefixes like "Answer:" or "The answer is:"
    predicted_answer = re.sub(r"^(Answer|The answer is|I think|Based on the document|According to the document):?\s*", "", predicted_answer)
    # Remove any trailing numbers or punctuation patterns
    predicted_answer = re.sub(r"([0-9]+\.?)+$", "", predicted_answer)
    predicted_answer = re.sub(r"1+$", "", predicted_answer)
    

    # Calculate generation time
    gen_time = time.time() - start_time
    
    # Compute metrics
    em_score = compute_exact_match(gold_answer, predicted_answer)
    f1_score = compute_f1(gold_answer, predicted_answer)
    qa_score = compute_qa_score(gold_answer, predicted_answer)
    
    # Print results
    print(f"Generation time: {gen_time:.2f} seconds")
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Gold Answer: {gold_answer}")
    print(f"Exact Match (EM): {em_score:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Combined QA Score: {qa_score:.2f}")
    
    return {
        "predicted_answer": predicted_answer,
        "gold_answer": gold_answer,
        "em_score": em_score,
        "f1_score": f1_score,
        "qa_score": qa_score,
        "generation_time": gen_time
    }


if __name__ == "__main__":
    # Import dataset library
    from datasets import load_dataset
    import random
    import numpy as np
    
    # Set random seed for reproducibility
    # only helpful when using actual dataset
    random.seed(42)
    np.random.seed(42)
    
    # Load NarrativeQA dataset with document summaries
    print("Loading NarrativeQA dataset...")
    dataset = load_dataset("deepmind/narrativeqa", split="validation")
    
    # Number of samples to evaluate
    num_samples = 10
    print(f"\nEvaluating {num_samples} NarrativeQA samples…")

    # rebuild sample_docs from the HF dataset
    sample_docs = []
    for ex in dataset.select(range(num_samples)):
        sample_docs.append({
            "document": ex["document"]["text"],           # full text lives here
            "question": ex["question"]["text"],           # question text
            "answer": [a["text"] for a in ex["answers"]][0] # list of all valid answers (first answer)
        })
    
    # Initialize metrics collection
    all_results = []
    
    # Process each sample
    for i, sample in enumerate(sample_docs):
        # Extract document, question, and answer
        doc_text = sample["document"]
        question = sample["question"]
        gold_answer = sample["answer"]
        
        # Tokenize document
        doc_tokens = tokenizer.encode(doc_text)
        
        print("\n" + "=" * 50)
        print(f"Sample {i+1}/{num_samples}")
        print(f"Document length: {len(doc_tokens)} tokens")
        print(f"Question: {question}")
        print(f"Gold Answer: {gold_answer}")
        print("=" * 50)
        
        # Run verification
        result = verify_upper_bound(doc_text, question, gold_answer)
        if result:
            all_results.append(result)
    
    # Calculate and print aggregate statistics
    if all_results:
        print("\n" + "=" * 50)
        print("AGGREGATE STATISTICS")
        print("=" * 50)
        
        em_scores = [r["em_score"] for r in all_results]
        f1_scores = [r["f1_score"] for r in all_results]
        qa_scores = [r["qa_score"] for r in all_results]
        gen_times = [r["generation_time"] for r in all_results]
        
        print(f"Samples evaluated: {len(all_results)}")
        print(f"Average Exact Match: {np.mean(em_scores):.4f} (std: {np.std(em_scores):.4f})")
        print(f"Average F1 Score: {np.mean(f1_scores):.4f} (std: {np.std(f1_scores):.4f})")
        print(f"Average QA Score: {np.mean(qa_scores):.4f} (std: {np.std(qa_scores):.4f})")
        print(f"Average generation time: {np.mean(gen_times):.2f} seconds")
        
        print("\nThis represents the upper-bound performance of the LLM with full document context.")
        print("Compare this to the SPARC agent's performance with limited context.")
