import sys
import os
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm
import numpy as np
import torch
from stable_baselines3 import PPO
from data.data_registry import get_dataset_iterator
from utils.llm_interface import LLMInterface
from transformers import AutoTokenizer
from utils.metrics import compute_qa_score_multi
from utils.model_paths import llama_32_3b, llama_31_8b
from envs.streaming_qagym import StreamingQAGym
import gymnasium as gym
from gymnasium import spaces

class SPARCPredictor:
    def __init__(self, model_path, env_kwargs):
        # Remove .zip extension if present to prevent double extension
        if model_path.endswith('.zip'):
            model_path = model_path[:-4]
        self.model = PPO.load(model_path)
        self.env = StreamingQAGym(**env_kwargs)
        
        # No need to patch the observation space - the default of 96 is what the model expects
        # This was previously incorrectly trying to patch to 48
        # The StreamingQAGym already sets question_max_len=96 by default
        
        self.tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B", use_fast=True)
        self.selected_idxs = []  # Store selected indices for metrics tracking

    def predict(self, question: str, ctx_chunks: list) -> str:
        obs, _ = self.env.reset()
        done = False
        self.selected_idxs = []
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(action)
            if action == 1:
                self.selected_idxs.append(self.env.chunk_idx - 1)
        selected = [ctx_chunks[i] for i in self.selected_idxs if i < len(ctx_chunks)]
        return " ".join(selected)

def evaluate_vanilla(dataset, n_eval=500, model_path=llama_31_8b, split="validation", chunk_size=256, max_ctx=2048, max_tokens=32, verbose=False, config=None):
    """Evaluate the vanilla LLM without SPARC."""
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B", use_fast=True)
    try:
        it = get_dataset_iterator(dataset, tok, split=split, chunk_size=chunk_size, config=config)
    except Exception as e:
        print(f"[ERROR] Could not create dataset iterator: {e}")
        return []
    
    llm = LLMInterface(model_path=model_path, n_ctx=max_ctx+50)
    scores = []
    
    for i, ex in enumerate(tqdm.tqdm(it, total=n_eval)):
        ctx = sum(ex["doc_chunks"], [])[:max_ctx]
        ctx_text = tok.decode(ctx, skip_special_tokens=True)
        q_text = tok.decode(ex["question_ids"], skip_special_tokens=True)
        
        prompt_text = f"Answer question using context in fewest words. Context: {ctx_text}\n\nQuestion: {q_text}\nAnswer:"
        answer = llm.generate_text(prompt_text, max_tokens=max_tokens, temperature=0.0)
        
        scores.append(compute_qa_score_multi(ex["answers"], answer))
        
        if verbose and i == 0:
            print("\n[VANILLA] Example 0:")
            print("Q:", q_text)
            print("Answer:", answer)
            print("References:", ex["answers"])
            print("Score:", scores[-1])
            
        if len(scores) == n_eval:
            break
            
    return scores

def evaluate_joint(dataset, n_eval=500, model_path=llama_31_8b, sparc_checkpoint="checkpoints/sparc_best_model_05_08.zip", split="validation", chunk_size=256, max_ctx=2048, max_tokens=32, verbose=False, config=None):
    """Evaluate both vanilla and SPARC LLMs on the same examples."""
    import time
    
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B", use_fast=True)
    try:
        # Use the provided config for the main data iterator
        it = get_dataset_iterator(dataset, tok, split=split, chunk_size=chunk_size, config=config)
    except Exception as e:
        print(f"[ERROR] Could not create dataset iterator: {e}")
        return [], []
    
    # SPARC setup
    def data_loader_fn():
        # Also pass the config to SPARC's internal data loader
        return get_dataset_iterator(dataset, tok, split=split, chunk_size=chunk_size, config=config)
    
    env_kwargs = {
        "data_loader_fn": data_loader_fn,
        "chunk_size": chunk_size,
        "max_window": max_ctx,
    }
    # Create the SPARC predictor
    sparc = SPARCPredictor(sparc_checkpoint, env_kwargs)
    llm = LLMInterface(model_path=model_path, n_ctx=max_ctx+50)
    
    # Track metrics
    vanilla_scores = []
    sparc_scores = []
    vanilla_token_counts = []  # Track token counts for vanilla prompts
    sparc_token_counts = []    # Track token counts for SPARC prompts
    vanilla_proc_times = []    # Track processing times for vanilla
    sparc_proc_times = []      # Track processing times for SPARC
    chunks_kept_counts = []    # Track how many chunks SPARC kept
    chunks_total_counts = []   # Track total available chunks
    vanilla_context_lengths = [] # Character lengths of vanilla contexts
    sparc_context_lengths = []   # Character lengths of SPARC contexts
    
    for i, ex in enumerate(tqdm.tqdm(it, total=n_eval)):
        ctx_chunks = [tok.decode(chunk, skip_special_tokens=True) for chunk in ex["doc_chunks"]]
        q_text = tok.decode(ex["question_ids"], skip_special_tokens=True)
        
        # Calculate total available chunks
        total_chunks = len(ctx_chunks)
        chunks_total_counts.append(total_chunks)
        
        # Vanilla evaluation
        full_context = ' '.join(ctx_chunks)
        vanilla_context_lengths.append(len(full_context))
        prompt_text = f"Answer question using context in fewest words. Context: {full_context}\n\nQuestion: {q_text}\nAnswer:"
        vanilla_token_count = len(tok.encode(prompt_text))
        vanilla_token_counts.append(vanilla_token_count)
        
        # Time the vanilla generation
        vanilla_start = time.time()
        vanilla_answer = llm.generate_text(prompt_text, max_tokens=max_tokens, temperature=0.0)
        vanilla_end = time.time()
        vanilla_proc_times.append(vanilla_end - vanilla_start)
        
        vanilla_scores.append(compute_qa_score_multi(ex["answers"], vanilla_answer))
        
        # SPARC evaluation
        # Track which chunks SPARC decides to keep
        sparc_start_select = time.time()
        selected_context = sparc.predict(q_text, ctx_chunks)
        sparc_end_select = time.time()
        
        # Count the number of chunks that were kept by SPARC
        # This is an approximation based on the length of the selected context
        sparc_context_lengths.append(len(selected_context))
        kept_chunks_estimate = len(sparc.selected_idxs) if hasattr(sparc, 'selected_idxs') else round(len(selected_context) / (len(full_context) / total_chunks))
        chunks_kept_counts.append(kept_chunks_estimate)
        
        prompt_text_sparc = f"Answer question using context in fewest words. Context: {selected_context}\n\nQuestion: {q_text}\nAnswer:"
        sparc_token_count = len(tok.encode(prompt_text_sparc))
        sparc_token_counts.append(sparc_token_count)
        
        # Time the SPARC generation
        sparc_start_gen = time.time()
        sparc_answer = llm.generate_text(prompt_text_sparc, max_tokens=max_tokens, temperature=0.0)
        sparc_end_gen = time.time()
        sparc_proc_times.append((sparc_end_select - sparc_start_select) + (sparc_end_gen - sparc_start_gen))
        
        sparc_scores.append(compute_qa_score_multi(ex["answers"], sparc_answer))
        
        if verbose and i == 0:
            print("\n[VANILLA] Example 0:")
            print("Q:", q_text)
            print(f"Context Length: {len(full_context)} chars, {vanilla_token_count} tokens")
            print(f"Processing Time: {vanilla_proc_times[-1]:.2f} seconds")
            print("Answer:", vanilla_answer)
            print("References:", ex["answers"])
            print("Score:", vanilla_scores[-1])
            
            print("\n[SPARC] Example 0:")
            print("Q:", q_text)
            print(f"Total Chunks: {total_chunks}, Kept Chunks: ~{kept_chunks_estimate} ({kept_chunks_estimate/total_chunks:.1%})")
            print(f"Context Length: {len(selected_context)} chars, {sparc_token_count} tokens ({len(selected_context)/len(full_context):.1%} of vanilla)")
            print(f"Selection Time: {sparc_end_select - sparc_start_select:.2f}s, Generation Time: {sparc_end_gen - sparc_start_gen:.2f}s")
            print("Selected Context (first 200 chars):", selected_context[:200] + "...")
            print("Answer:", sparc_answer)
            print("References:", ex["answers"])
            print("Score:", sparc_scores[-1])
        
        if len(vanilla_scores) == n_eval:
            break
    
    # Compile all metrics to return
    metrics = {
        "vanilla_scores": vanilla_scores,
        "sparc_scores": sparc_scores,
        "vanilla_token_counts": vanilla_token_counts,
        "sparc_token_counts": sparc_token_counts,
        "vanilla_proc_times": vanilla_proc_times,
        "sparc_proc_times": sparc_proc_times,
        "chunks_kept_counts": chunks_kept_counts,
        "chunks_total_counts": chunks_total_counts,
        "vanilla_context_lengths": vanilla_context_lengths,
        "sparc_context_lengths": sparc_context_lengths
    }
            
    return vanilla_scores, sparc_scores, metrics

def evaluate_sparc(dataset, n_eval=500, model_path=llama_31_8b, sparc_checkpoint="checkpoints/sparc_3b_baseline.zip", 
                  split="validation", chunk_size=256, max_ctx=2048, max_tokens=32, verbose=False, config=None):
    """Evaluate the LLM with SPARC-based context selection."""
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B", use_fast=True)
    try:
        it = get_dataset_iterator(dataset, tok, split=split, chunk_size=chunk_size, config=config)
    except Exception as e:
        print(f"[ERROR] Could not create dataset iterator: {e}")
        return []
    
    # Prepare SPARC model with gym environment
    def data_loader_fn():
        return get_dataset_iterator(dataset, tok, split=split, chunk_size=chunk_size, config=config)
    
    env_kwargs = {
        "data_loader_fn": data_loader_fn,
        "chunk_size": chunk_size,
        "max_window": max_ctx,
    }
    sparc_predictor = SPARCPredictor(sparc_checkpoint, env_kwargs)
    llm = LLMInterface(model_path=model_path, n_ctx=max_ctx+50)
    scores = []
    
    for i, ex in enumerate(tqdm.tqdm(it, total=n_eval)):
        ctx_chunks = [tok.decode(chunk, skip_special_tokens=True) for chunk in ex["doc_chunks"]]
        q_text = tok.decode(ex["question_ids"], skip_special_tokens=True)
        
        # Generate answer using SPARC-selected context
        selected_context = sparc_predictor.predict(q_text, ctx_chunks)
        prompt_text = f"Answer question using context in fewest words. Context: {selected_context}\n\nQuestion: {q_text}\nAnswer:"
        answer = llm.generate_text(prompt_text, max_tokens=max_tokens, temperature=0.0)
        
        scores.append(compute_qa_score_multi(ex["answers"], answer))
        
        if verbose and i == 0:
            print("\n[SPARC] Example 0:")
            print("Q:", q_text)
            print("Selected Context (first 200 chars):", selected_context[:200] + "...")
            print("Answer:", answer)
            print("References:", ex["answers"])
            print("Score:", scores[-1])
            
        if len(scores) == n_eval:
            break
            
    return scores

def get_datasets_from_config(config_path="conf/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    datasets = config.get("data", {}).get("dataset_order")
    if not datasets:
        datasets = config.get("data", {}).get("datasets", [])
    return datasets

if __name__ == "__main__":
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description="Compare vanilla LLM reader with SPARC-augmented version.")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated list of datasets to evaluate (default: all from config)")
    parser.add_argument("--n_eval", type=int, default=5, help="Number of examples to evaluate per dataset")
    parser.add_argument("--model", type=str, default=llama_31_8b, help="Path to LLM weights (GGUF)")
    parser.add_argument("--sparc_checkpoint", type=str, default="checkpoints/sparc_best_model_05_08_25.zip", 
                       help="Path to SPARC checkpoint")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--chunk_size", type=int, default=256, help="Document chunk size")
    parser.add_argument("--max_ctx", type=int, default=2048, help="Max context tokens")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max answer tokens")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output for first example")
    args = parser.parse_args()
    
    # Get list of datasets to evaluate
    if args.datasets:
        datasets_to_eval = args.datasets.split(',')
    else:
        # Get all datasets from config
        datasets_to_eval = get_datasets_from_config()
    
    print(f"\n=== Evaluating on {len(datasets_to_eval)} datasets: {', '.join(datasets_to_eval)} ===")
    
    # Store results for all datasets
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets_to_eval:
        print(f"\n{'='*20} Dataset: {dataset_name} {'='*20}")
        
        # For NarrativeQA, ensure we use full documents not summaries
        # Create a config to pass to evaluate_joint
        dataset_config = None
        if dataset_name == "narrativeqa":
            # Explicitly set use_summaries=False for NarrativeQA
            dataset_config = {"narrativeqa": {"use_summaries": False}}
            print("Using FULL DOCUMENTS for NarrativeQA (use_summaries=False)")
        
        # Run evaluation for this dataset
        try:
            vanilla_scores, sparc_scores, metrics = evaluate_joint(
                dataset=dataset_name,
                n_eval=args.n_eval,
                model_path=args.model,
                sparc_checkpoint=args.sparc_checkpoint,
                split=args.split,
                chunk_size=args.chunk_size,
                max_ctx=args.max_ctx,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
                config=dataset_config  # Pass the dataset-specific config
            )
            
            # Store all results and metrics
            all_results[dataset_name] = {
                "vanilla": vanilla_scores,
                "sparc": sparc_scores,
                "metrics": metrics
            }
            
            # Print comparison for this dataset
            if len(vanilla_scores) > 0 and len(sparc_scores) > 0:
                # Calculate averages of all metrics
                vanilla_mean = np.mean(vanilla_scores)
                sparc_mean = np.mean(sparc_scores)
                
                # Token usage metrics
                vanilla_tokens_avg = np.mean(metrics['vanilla_token_counts'])
                sparc_tokens_avg = np.mean(metrics['sparc_token_counts'])
                token_reduction = 1 - (sparc_tokens_avg / vanilla_tokens_avg) if vanilla_tokens_avg > 0 else 0
                
                # Processing time metrics
                vanilla_time_avg = np.mean(metrics['vanilla_proc_times'])
                sparc_time_avg = np.mean(metrics['sparc_proc_times'])
                time_change = (sparc_time_avg / vanilla_time_avg) - 1 if vanilla_time_avg > 0 else 0
                
                # Chunk selection metrics
                chunks_kept_avg = np.mean(metrics['chunks_kept_counts'])
                chunks_total_avg = np.mean(metrics['chunks_total_counts'])
                chunk_keep_ratio = chunks_kept_avg / chunks_total_avg if chunks_total_avg > 0 else 0
                
                # Context length metrics
                vanilla_ctx_len_avg = np.mean(metrics['vanilla_context_lengths'])
                sparc_ctx_len_avg = np.mean(metrics['sparc_context_lengths'])
                ctx_reduction = 1 - (sparc_ctx_len_avg / vanilla_ctx_len_avg) if vanilla_ctx_len_avg > 0 else 0
                
                # Print detailed results
                print(f"\n--- {dataset_name} Results ---")
                print(f"Vanilla LLM - Average QA Score: {vanilla_mean:.4f} ± {np.std(vanilla_scores):.4f}")
                print(f"SPARC-Augmented - Average QA Score: {sparc_mean:.4f} ± {np.std(sparc_scores):.4f}")
                
                # Calculate QA score improvement
                if vanilla_mean > 0:
                    improvement = (sparc_mean - vanilla_mean) / vanilla_mean * 100
                    print(f"QA Score Improvement: {improvement:+.2f}%")
                else:
                    print("Cannot calculate percentage improvement (vanilla score is zero)")
                
                # Print efficiency metrics
                print("\n--- Efficiency Metrics ---")
                print(f"Token Usage: Vanilla={vanilla_tokens_avg:.1f}, SPARC={sparc_tokens_avg:.1f}, Reduction={token_reduction:.1%}")
                print(f"Context Length: Vanilla={vanilla_ctx_len_avg:.1f} chars, SPARC={sparc_ctx_len_avg:.1f} chars, Reduction={ctx_reduction:.1%}")
                print(f"Processing Time: Vanilla={vanilla_time_avg:.2f}s, SPARC={sparc_time_avg:.2f}s, Change={time_change:+.1%}")
                print(f"Chunk Selection: Kept {chunks_kept_avg:.1f}/{chunks_total_avg:.1f} chunks ({chunk_keep_ratio:.1%})")
            else:
                print(f"Warning: No valid scores obtained for {dataset_name}")
        
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue
    
    # Print overall summary if multiple datasets were evaluated
    if len(all_results) > 1:
        print("\n" + "="*50)
        print("OVERALL SUMMARY ACROSS ALL DATASETS")
        print("="*50)
        
        # Calculate overall average scores
        all_vanilla = [score for dataset in all_results.values() for score in dataset["vanilla"]]
        all_sparc = [score for dataset in all_results.values() for score in dataset["sparc"]]
        
        # Combine metrics across datasets
        all_metrics = {
            'vanilla_token_counts': [],
            'sparc_token_counts': [],
            'vanilla_proc_times': [],
            'sparc_proc_times': [],
            'chunks_kept_counts': [],
            'chunks_total_counts': [],
            'vanilla_context_lengths': [],
            'sparc_context_lengths': []
        }
        
        # Gather all metrics across datasets
        for dataset in all_results.values():
            if 'metrics' in dataset:
                for metric_name, metric_values in dataset['metrics'].items():
                    if metric_name in all_metrics:
                        all_metrics[metric_name].extend(metric_values)
        
        if all_vanilla and all_sparc:
            # QA Score metrics
            vanilla_overall = np.mean(all_vanilla)
            sparc_overall = np.mean(all_sparc)
            
            print(f"Vanilla LLM - Overall QA Score: {vanilla_overall:.4f} ± {np.std(all_vanilla):.4f}")
            print(f"SPARC-Augmented - Overall QA Score: {sparc_overall:.4f} ± {np.std(all_sparc):.4f}")
            
            # Calculate overall QA improvement
            if vanilla_overall > 0:
                overall_improvement = (sparc_overall - vanilla_overall) / vanilla_overall * 100
                print(f"\nOverall QA Score Improvement: {overall_improvement:+.2f}%")
            
            # Calculate efficiency metrics if available
            if all(len(all_metrics[metric]) > 0 for metric in all_metrics):
                # Token usage metrics
                vanilla_tokens_avg = np.mean(all_metrics['vanilla_token_counts'])
                sparc_tokens_avg = np.mean(all_metrics['sparc_token_counts'])
                token_reduction = 1 - (sparc_tokens_avg / vanilla_tokens_avg) if vanilla_tokens_avg > 0 else 0
                
                # Processing time metrics
                vanilla_time_avg = np.mean(all_metrics['vanilla_proc_times'])
                sparc_time_avg = np.mean(all_metrics['sparc_proc_times'])
                time_change = (sparc_time_avg / vanilla_time_avg) - 1 if vanilla_time_avg > 0 else 0
                
                # Chunk selection metrics
                chunks_kept_avg = np.mean(all_metrics['chunks_kept_counts'])
                chunks_total_avg = np.mean(all_metrics['chunks_total_counts'])
                chunk_keep_ratio = chunks_kept_avg / chunks_total_avg if chunks_total_avg > 0 else 0
                
                # Context length metrics
                vanilla_ctx_len_avg = np.mean(all_metrics['vanilla_context_lengths'])
                sparc_ctx_len_avg = np.mean(all_metrics['sparc_context_lengths'])
                ctx_reduction = 1 - (sparc_ctx_len_avg / vanilla_ctx_len_avg) if vanilla_ctx_len_avg > 0 else 0
                
                print("\n--- Overall Efficiency Metrics ---")
                print(f"Token Usage: Vanilla={vanilla_tokens_avg:.1f}, SPARC={sparc_tokens_avg:.1f}, Reduction={token_reduction:.1%}")
                print(f"Context Length: Vanilla={vanilla_ctx_len_avg:.1f} chars, SPARC={sparc_ctx_len_avg:.1f} chars, Reduction={ctx_reduction:.1%}")
                print(f"Processing Time: Vanilla={vanilla_time_avg:.2f}s, SPARC={sparc_time_avg:.2f}s, Change={time_change:+.1%}")
                print(f"Chunk Selection: Kept {chunks_kept_avg:.1f}/{chunks_total_avg:.1f} chunks ({chunk_keep_ratio:.1%})")
            
            # Final assessment with focus on token efficiency
            print("\n--- Overall Assessment ---")
            if overall_improvement > 0:
                print(f"✓ SPARC improved QA performance across datasets! ({overall_improvement:+.2f}%) 🎉")
            else:
                print(f"✗ SPARC did not improve QA performance. ({overall_improvement:.2f}%)")
                
            if token_reduction > 0.3:  # If SPARC reduces tokens by more than 30%
                print(f"✓ SPARC significantly reduced token usage by {token_reduction:.1%} 💰")
                print(f"  This represents substantial potential cost savings for API-based LLMs.")
            elif token_reduction > 0:
                print(f"✓ SPARC reduced token usage by {token_reduction:.1%}")
            else:
                print(f"✗ SPARC did not reduce token usage ({token_reduction:.1%})")
        else:
            print("Not enough valid results to calculate overall scores.")
