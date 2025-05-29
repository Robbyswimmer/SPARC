#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPARC Evaluation Script

This script evaluates trained RL models for streaming context management by comparing:
1. Baseline LLM (using all available context)
2. SPARC-enhanced LLM (using RL-based context management)
3. ***Sliding Window LLM (using sliding window context management) FIXME
4. ***TFIDF LLM (using TFIDF-based context management) FIXME

Results are analyzed across multiple metrics including QA accuracy, token efficiency,
and context management behavior.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
import torch
import wandb
from omegaconf import OmegaConf, DictConfig

from envs.streaming_qagym import StreamingQAGym
from data.data_registry import CurriculumDataLoader
from transformers import AutoTokenizer


def calculate_improvement(metric: str, baseline_value: float, sparc_value: float) -> float:
    """
    Calculate the percentage improvement of SPARC over the baseline for a given metric.
    
    Args:
        metric: The metric name
        baseline_value: The baseline value
        sparc_value: The SPARC value
        
    Returns:
        The percentage improvement (positive is better, negative is worse)
    """
    if baseline_value == 0:
        return float('inf') if sparc_value > 0 else 0.0
    
    # For tokens_used, lower is better
    if metric == "tokens_used":
        # Calculate how much fewer tokens were used (positive % means token reduction)
        return (baseline_value - sparc_value) / baseline_value * 100.0
    else:
        # For other metrics (qa_score, em_score, f1_score, token_efficiency), higher is better
        return (sparc_value - baseline_value) / baseline_value * 100.0


# Correct baseline implementation that properly keeps all tokens
class StreamingQAGymFull(StreamingQAGym):
    """Version of StreamingQAGym that keeps ALL chunks, regardless of constraints.
    
    This baseline implementation represents a traditional LLM that uses all available context
    without any pruning. It always performs the KEEP action (action 1) to ensure all tokens
    are included in the context.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "baseline_full"
        # Store the original max_window but set it to a very high value
        self.original_max_window = self.max_window
        self.max_window = 1000000  # Set an extremely high window size
    
    def step(self, action):
        """Override step to always KEEP all chunks.
        
        In StreamingQAGym, action 1 is KEEP (not action 0).        
        This override ensures we always use KEEP action regardless of what's passed in.
        """
        # Always use KEEP (action 1) and ensure window constraint doesn't apply
        # due to our very large max_window setting
        return super().step(1)  # 1 = KEEP, not 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SPARC models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="conf/config.yaml", 
        help="Path to config file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="checkpoints/sparc_best_model_05_08_25.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=100, 
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs="*",
        default=["narrativeqa"], 
        help="Specific datasets to evaluate on (default: use all from config)"
    )
    parser.add_argument(
        "--per_dataset_episodes", 
        type=int, 
        default=10, 
        help="Number of episodes to evaluate per dataset"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Log to Weights & Biases"
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_evaluation_episode(
    env: gym.Env, 
    model: Optional[PPO] = None,
    deterministic: bool = True,
    seed: Optional[int] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run a single evaluation episode using either the provided model or baseline.
    
    Args:
        env: The environment to run the episode in
        model: The RL model to use (if None, will use baseline behavior)
        deterministic: Whether to use deterministic actions
        seed: Random seed for episode reproducibility
        
    Returns:
        episode_metrics: Dict of episode-level metrics
        step_data: List of step-level data
    """
    # In Gymnasium, seed is passed to reset
    obs, _ = env.reset(seed=seed)
    done = False
    step_data = []
    
    # Episode metrics
    total_reward = 0
    keep_count = 0
    drop_count = 0
    compress_count = 0
    
    while not done:
        # Get action
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            # For baseline, always KEEP (which is action 1 in StreamingQAGym)
            action = 1  # KEEP action (action 1 in StreamingQAGym)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Track more detailed information from the step result
        chunk_was_kept = info.get("chunk_was_kept", False)
        tokens_in_chunk = len(info.get("token_ids", []))
        
        # Record step data with enhanced information
        step_info = {
            "action": int(action),
            "reward": float(reward),
            "token_ids": info.get("token_ids", []),
            "keep_heuristic": info.get("keep_heuristic", 0),
            "chunk_was_kept": chunk_was_kept,
            "tokens_in_chunk": tokens_in_chunk
        }
        step_data.append(step_info)
        
        # Update metrics based on attempted actions
        total_reward += reward
        if action == 1:  # KEEP - action 1 is KEEP in StreamingQAGym
            keep_count += 1
            # We can optionally track successful keep count if info provides it
            if "chunk_was_kept" in info:
                if info["chunk_was_kept"]:
                    step_info["successful_keep"] = True
                else:
                    step_info["successful_keep"] = False  # KEEP attempted but failed (e.g., window overflow)
        elif action == 0:  # DROP - action 0 is DROP in StreamingQAGym
            drop_count += 1
        elif action == 2:  # COMPRESS (if implemented)
            compress_count += 1
        
        # Update observation
        obs = next_obs
    
    # Get final metrics from info
    qa_score = info.get("qa_score", 0)
    tokens_used = info.get("tokens_used", 0)
    em_score = info.get("max_em", info.get("exact_match", 0))
    f1_score = info.get("max_f1", info.get("f1", 0))
    
    # Count successful keeps from step data if available
    successful_keep_count = sum(1 for s in step_data if s.get("successful_keep", False))
    
    # Get total document tokens seen directly from the environment
    total_doc_tokens_seen = info.get("total_doc_tokens_seen", 0)

    # Get tokens used in the final prompt and number of question tokens
    question_token_ids = info.get("question_token_ids", [])
    num_question_tokens = len(question_token_ids)

    # Get document tokens actually kept in the context window at QA time
    doc_tokens_in_final_prompt = info.get("doc_tokens_in_final_prompt", 0)

    # Calculate Tokens Kept Ratio: (doc tokens in final prompt) / (total doc tokens streamed)
    tokens_kept_ratio = doc_tokens_in_final_prompt / total_doc_tokens_seen if total_doc_tokens_seen > 0 else 0

    # Debug print for verification
    print(f"[DEBUG] tokens_used_final_prompt: {tokens_used}, doc_tokens_in_final_prompt: {doc_tokens_in_final_prompt}, total_doc_tokens_seen: {total_doc_tokens_seen}, num_question_tokens: {num_question_tokens}, tokens_kept_ratio: {tokens_kept_ratio:.4f}")

    # Store metrics
    # QA score based on the first gold answer (for backward compatibility with some older logs)
    episode_metrics = {
        "qa_score": qa_score, # Uses max_over_refs if available, else single ref
        "em_score": em_score, # Single reference
        "f1_score": f1_score, # Single reference
        "max_em_score": info.get("max_em", info.get("exact_match", 0)), # Max over references
        "max_f1_score": info.get("max_f1", info.get("f1", 0)), # Max over references
        "tokens_used": tokens_used,
        "keep_count": info.get("keep_count", 0),
        "drop_count": info.get("drop_count", 0),
        "compress_count": info.get("compress_count", 0),
        "keep_ratio": keep_count / (keep_count + drop_count + compress_count) if (keep_count + drop_count + compress_count) > 0 else 0,
        "successful_keep_ratio": successful_keep_count / (keep_count + drop_count + compress_count) if (keep_count + drop_count + compress_count) > 0 else 0,
        "tokens_kept_ratio": tokens_kept_ratio,  # More accurate metric of tokens actually kept
        "total_chunks": keep_count + drop_count + compress_count,
        "total_tokens_seen": total_doc_tokens_seen, # This is total *document* tokens streamed
        "token_efficiency": qa_score / tokens_used if tokens_used > 0 else 0,
        # Adding raw values for deeper analysis if needed
        "raw_doc_tokens_in_final_prompt": doc_tokens_in_final_prompt,
        "raw_total_doc_tokens_seen": total_doc_tokens_seen,
        "raw_num_question_tokens": num_question_tokens
    }

    return episode_metrics, step_data


def evaluate_model(
    model_path: str,
    cfg: DictConfig,
    num_episodes: int = 50,
    datasets: List[str] = None,
    per_dataset_episodes: int = 10,
    split: str = "test",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a trained model against baseline on multiple episodes.
    
    Args:
        model_path: Path to the trained model
        cfg: Configuration
        num_episodes: Number of episodes to evaluate
        dataset: Dataset to evaluate on
        split: Dataset split to use
        seed: Random seed
        
    Returns:
        results: Dictionary containing evaluation results
    """
    # Set random seed
    set_seed(seed)
    
    # Load the trained model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Get the tokenizer used by StreamingQAGym
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B",
        use_fast=True,
        add_special_tokens=True
    )
    # Fix for pad_token_id being None
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Use datasets from config if none specified
    if datasets is None:
        if "datasets" in cfg.data:
            datasets = cfg.data.datasets
        else:
            datasets = ["narrativeqa"]  # Default fallback
    
    print(f"\nEvaluating on datasets: {datasets}")
    
    # Results storage - now organized by dataset
    all_results = {
        "sparc": {
            "episodes": [],
            "step_data": [],
            "metrics": defaultdict(list),
            "per_dataset": {}
        },
        "baseline": {
            "episodes": [],
            "step_data": [],
            "metrics": defaultdict(list),
            "per_dataset": {}
        },
        "improvements": {},
        "per_dataset_improvements": {},
    }
    
    # Initialize per-dataset metrics
    for dataset_name in datasets:
        all_results["sparc"]["per_dataset"][dataset_name] = {
            "episodes": [],
            "metrics": defaultdict(list)
        }
        all_results["baseline"]["per_dataset"][dataset_name] = {
            "episodes": [],
            "metrics": defaultdict(list)
        }
    
    # For each dataset, run evaluation
    for dataset_name in datasets:
        print(f"\n{'-'*40}\nEvaluating dataset: {dataset_name}\n{'-'*40}")
        
        # Use validation cache if available
        validation_cache_path = None
        if "validation_cache_dir" in cfg.data and os.path.isdir(cfg.data.validation_cache_dir):
            dataset_cache_dir = os.path.join(cfg.data.validation_cache_dir, dataset_name, split)
            if os.path.isdir(dataset_cache_dir):
                validation_cache_path = dataset_cache_dir
                print(f"Using cached validation data for {dataset_name} from {validation_cache_path}")
            else:
                os.makedirs(dataset_cache_dir, exist_ok=True)
                print(f"Created cache directory for {dataset_name}: {dataset_cache_dir}")
        
        # Create curriculum data loader for this dataset
        curriculum = CurriculumDataLoader(
            tokenizer=tokenizer,
            dataset_order=[dataset_name],  # Only use the current dataset
            qa_thresholds=[1.0],     # Threshold that will never be reached (keep just one dataset)
            split=split,
            seed=seed,
            chunk_size=256,
            dataset_config={
                dataset_name: {
                    "validation_cache_path": validation_cache_path
                }
            }  # Pass validation_cache_path through dataset_config instead
        )
        
        # Get the data loader function
        data_loader_fn = curriculum.get_data_loader()
        
        # Create environments with identical settings for this dataset
        env_kwargs = {
            "dataset_name": dataset_name,
            "split": split,
            "max_window": cfg.env.max_window,
            "alpha": cfg.env.reward.alpha,
            "beta_keep": cfg.env.reward.beta_keep,
            "gamma_step": cfg.env.reward.gamma_step,
            "kappa": cfg.env.reward.kappa,
            "use_hindsight": cfg.env.reward.use_hindsight,
            "data_loader_fn": data_loader_fn,  # Use the dataset-specific loader
            "seed": seed + datasets.index(dataset_name),  # Unique seed per dataset
        }
        
        # Create environments with independent iterators
        sparc_env = StreamingQAGym(**env_kwargs)
        
        # Create a fresh data loader function for the baseline to prevent iterator sharing issues
        baseline_data_loader_fn = curriculum.get_data_loader()  # Get a fresh iterator
        baseline_env_kwargs = env_kwargs.copy()
        baseline_env_kwargs["data_loader_fn"] = baseline_data_loader_fn
        baseline_env = StreamingQAGymFull(**baseline_env_kwargs)
            
        # Calculate episodes for this dataset
        dataset_episodes = min(per_dataset_episodes, num_episodes)
            
        # Run evaluation episodes for this dataset
        print(f"Evaluating on {dataset_episodes} episodes for {dataset_name}...")
        
        # SPARC evaluation for this dataset
        print(f"Evaluating SPARC model on {dataset_name}...")
        for i in tqdm(range(dataset_episodes)):
            # Use a different seed for each episode, but make it deterministic across systems
            episode_seed = seed + datasets.index(dataset_name)*1000 + i
            episode_metrics, step_data = run_evaluation_episode(sparc_env, model, seed=episode_seed)
            
            # Add dataset name to metrics
            episode_metrics["dataset"] = dataset_name
            
            # Add to overall results
            all_results["sparc"]["episodes"].append(episode_metrics)
            all_results["sparc"]["step_data"].append(step_data)
            
            # Add to per-dataset results
            all_results["sparc"]["per_dataset"][dataset_name]["episodes"].append(episode_metrics)
            
            # Store each metric for later statistics (both overall and per-dataset)
            for key, value in episode_metrics.items():
                all_results["sparc"]["metrics"][key].append(value)
                all_results["sparc"]["per_dataset"][dataset_name]["metrics"][key].append(value)
        
        # Baseline evaluation for this dataset
        print(f"Evaluating baseline approach on {dataset_name}...")
        for i in tqdm(range(dataset_episodes)):
            # Use the same seeds as the SPARC evaluation for fair comparison
            episode_seed = seed + datasets.index(dataset_name)*1000 + i
            episode_metrics, step_data = run_evaluation_episode(baseline_env, None, seed=episode_seed)
            
            # Add dataset name to metrics
            episode_metrics["dataset"] = dataset_name
            
            # Add to overall results
            all_results["baseline"]["episodes"].append(episode_metrics)
            all_results["baseline"]["step_data"].append(step_data)
            
            # Add to per-dataset results
            all_results["baseline"]["per_dataset"][dataset_name]["episodes"].append(episode_metrics)
            
            # Store each metric for later statistics (both overall and per-dataset)
            for key, value in episode_metrics.items():
                all_results["baseline"]["metrics"][key].append(value)
                all_results["baseline"]["per_dataset"][dataset_name]["metrics"][key].append(value)
    
    # Compute aggregate statistics for overall results
    for system in ["sparc", "baseline"]:
        for metric, values in all_results[system]["metrics"].items():
            if isinstance(values[0], (int, float)) and metric != "dataset":  # Skip non-numeric or dataset metrics
                all_results[system][f"{metric}_mean"] = np.mean(values)
                all_results[system][f"{metric}_std"] = np.std(values)
                all_results[system][f"{metric}_min"] = np.min(values)
                all_results[system][f"{metric}_max"] = np.max(values)
                all_results[system][f"{metric}_median"] = np.median(values)
        
        # Also compute per-dataset statistics
        for dataset_name in datasets:
            for metric, values in all_results[system]["per_dataset"][dataset_name]["metrics"].items():
                if isinstance(values[0], (int, float)) and metric != "dataset":
                    all_results[system]["per_dataset"][dataset_name][f"{metric}_mean"] = np.mean(values)
                    all_results[system]["per_dataset"][dataset_name][f"{metric}_std"] = np.std(values)
    
    # Calculate overall improvement percentages
    metrics_to_compare = [
        "qa_score", "em_score", "f1_score", "tokens_used", "token_efficiency"
    ]
    
    # Overall improvements
    for metric in metrics_to_compare:
        baseline_value = all_results["baseline"].get(f"{metric}_mean", 0)
        sparc_value = all_results["sparc"].get(f"{metric}_mean", 0)
        
        all_results["improvements"][metric] = calculate_improvement(metric, baseline_value, sparc_value)
    
    # Per-dataset improvements
    all_results["per_dataset_improvements"] = {}
    for dataset_name in datasets:
        all_results["per_dataset_improvements"][dataset_name] = {}
        for metric in metrics_to_compare:
            baseline_value = all_results["baseline"]["per_dataset"][dataset_name].get(f"{metric}_mean", 0)
            sparc_value = all_results["sparc"]["per_dataset"][dataset_name].get(f"{metric}_mean", 0)
            
            all_results["per_dataset_improvements"][dataset_name][metric] = calculate_improvement(metric, baseline_value, sparc_value)
    
    return all_results


def create_per_dataset_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame with per-dataset metrics.
    
    Args:
        results: The evaluation results with per-dataset information
        
    Returns:
        DataFrame with per-dataset metrics
    """
    data = []
    for dataset_name, metrics in results["per_dataset_improvements"].items():
        row = {
            "dataset": dataset_name,
            "sparc_qa_score": results["sparc"]["per_dataset"][dataset_name]["qa_score_mean"],
            "baseline_qa_score": results["baseline"]["per_dataset"][dataset_name]["qa_score_mean"],
            "qa_improvement": metrics["qa_score"],
            "sparc_tokens": results["sparc"]["per_dataset"][dataset_name]["tokens_used_mean"],
            "baseline_tokens": results["baseline"]["per_dataset"][dataset_name]["tokens_used_mean"],
            "token_reduction": metrics["tokens_used"],
            "sparc_token_efficiency": results["sparc"]["per_dataset"][dataset_name]["token_efficiency_mean"],
            "baseline_token_efficiency": results["baseline"]["per_dataset"][dataset_name]["token_efficiency_mean"],
            "efficiency_improvement": metrics["token_efficiency"],
            "keep_ratio": results["sparc"]["per_dataset"][dataset_name]["keep_ratio_mean"],
            "successful_keep_ratio": results["sparc"]["per_dataset"][dataset_name].get("successful_keep_ratio_mean", 0),
            "tokens_kept_ratio": results["sparc"]["per_dataset"][dataset_name].get("tokens_kept_ratio_mean", 0)
        }
        data.append(row)
    
    return pd.DataFrame(data)


def plot_per_dataset_comparison(results: Dict[str, Any], output_path: str):
    """
    Plot per-dataset comparison of metrics.
    
    Args:
        results: The evaluation results
        output_path: Path to save the plot
    """
    # Create dataframe
    df = create_per_dataset_dataframe(results)
    
    # Set up plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Per-Dataset Performance Comparison: SPARC vs Baseline", fontsize=16)
    
    # Plot QA scores
    ax = axes[0, 0]
    df_qa = df[["dataset", "sparc_qa_score", "baseline_qa_score"]].melt(
        id_vars=["dataset"], 
        value_vars=["sparc_qa_score", "baseline_qa_score"],
        var_name="system", 
        value_name="qa_score"
    )
    sns.barplot(x="dataset", y="qa_score", hue="system", data=df_qa, ax=ax)
    ax.set_title("QA Score Comparison")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("QA Score")
    for i, v in enumerate(df["qa_improvement"]):
        ax.text(i, max(df.loc[i, "sparc_qa_score"], df.loc[i, "baseline_qa_score"]) + 0.02, 
                f"{v:.1f}%", ha="center", fontweight="bold")
    
    # Plot token usage
    ax = axes[0, 1]
    df_tokens = df[["dataset", "sparc_tokens", "baseline_tokens"]].melt(
        id_vars=["dataset"], 
        value_vars=["sparc_tokens", "baseline_tokens"],
        var_name="system", 
        value_name="tokens_used"
    )
    sns.barplot(x="dataset", y="tokens_used", hue="system", data=df_tokens, ax=ax)
    ax.set_title("Token Usage Comparison")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Tokens Used")
    for i, v in enumerate(df["token_reduction"]):
        ax.text(i, max(df.loc[i, "sparc_tokens"], df.loc[i, "baseline_tokens"]) + 10, 
                f"{v:.1f}%", ha="center", fontweight="bold")
    
    # Plot token efficiency
    ax = axes[1, 0]
    df_eff = df[["dataset", "sparc_token_efficiency", "baseline_token_efficiency"]].melt(
        id_vars=["dataset"], 
        value_vars=["sparc_token_efficiency", "baseline_token_efficiency"],
        var_name="system", 
        value_name="token_efficiency"
    )
    sns.barplot(x="dataset", y="token_efficiency", hue="system", data=df_eff, ax=ax)
    ax.set_title("Token Efficiency Comparison")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Token Efficiency (QA Score / Token)")
    for i, v in enumerate(df["efficiency_improvement"]):
        ax.text(i, max(df.loc[i, "sparc_token_efficiency"], df.loc[i, "baseline_token_efficiency"]) + 0.00005, 
                f"{v:.1f}%", ha="center", fontweight="bold")
    
    # Plot keep ratios
    ax = axes[1, 1]
    if "tokens_kept_ratio" in df.columns:
        sns.barplot(x="dataset", y="value", hue="variable", 
                   data=df[["dataset", "keep_ratio", "successful_keep_ratio", "tokens_kept_ratio"]].melt(id_vars=["dataset"]), 
                   ax=ax)
    else:
        sns.barplot(x="dataset", y="keep_ratio", data=df, ax=ax)
    ax.set_title("SPARC Keep Ratios")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Ratio")
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_results(results: Dict[str, Any]):
    """
    Print a summary of the evaluation results to the console.
    
    Args:
        results: The evaluation results
    """
    print("\n" + "=" * 50)
    print("OVERALL: SPARC vs Baseline:")
    metrics_to_report = [
        ("QA Score", "qa_score"),
        ("Exact Match", "em_score"),
        ("F1 Score", "f1_score"),
        ("Tokens Used", "tokens_used"),
        ("Token Efficiency", "token_efficiency"),
    ]
    
    for metric_name, metric_key in metrics_to_report:
        sparc_value = results["sparc"][f"{metric_key}_mean"]
        baseline_value = results["baseline"][f"{metric_key}_mean"]
        improvement = results["improvements"][metric_key]
        
        # Special formatting based on metric
        if metric_key in ["token_efficiency"]:
            print(f"  {metric_name}:\t {sparc_value:.6f} vs {baseline_value:.6f} ({improvement:+.1f}%)")
        else:
            print(f"  {metric_name}:\t {sparc_value:.4f} vs {baseline_value:.4f} ({improvement:+.1f}%)")
    
    keep_metrics = [
        ("Keep Ratio (attempted)", "keep_ratio"),
        ("Successful Keep Ratio", "successful_keep_ratio"),
        ("Tokens Kept Ratio", "tokens_kept_ratio")
    ]
    
    print("\nSPARC Action Metrics:")
    for metric_name, metric_key in keep_metrics:
        if f"{metric_key}_mean" in results["sparc"]:
            print(f"  {metric_name}: {results['sparc'][f'{metric_key}_mean']*100:.2f}%")
    
    # Print per-dataset summary if available
    if "per_dataset_improvements" in results:
        print("\n" + "=" * 50)
        print("PER-DATASET SUMMARY:")
        for dataset_name, improvements in results["per_dataset_improvements"].items():
            print(f"\n{dataset_name}:")
            for metric_name, metric_key in metrics_to_report:
                sparc_value = results["sparc"]["per_dataset"][dataset_name][f"{metric_key}_mean"]
                baseline_value = results["baseline"]["per_dataset"][dataset_name][f"{metric_key}_mean"]
                improvement = improvements[metric_key]
                
                # Special formatting based on metric
                if metric_key in ["token_efficiency"]:
                    print(f"  {metric_name}:\t {sparc_value:.6f} vs {baseline_value:.6f} ({improvement:+.1f}%)")
                else:
                    print(f"  {metric_name}:\t {sparc_value:.4f} vs {baseline_value:.4f} ({improvement:+.1f}%)")
            
            # Print dataset-specific keep ratios
            for metric_name, metric_key in keep_metrics:
                if f"{metric_key}_mean" in results["sparc"]["per_dataset"][dataset_name]:
                    print(f"  {metric_name}: {results['sparc']['per_dataset'][dataset_name][f'{metric_key}_mean']*100:.2f}%")
    
    print("=" * 50)


def plot_results(results: Dict[str, Any], output_dir: str):
    """
    Plot various visualizations for the evaluation results.
    
    Args:
        results: The evaluation results
        output_dir: Directory to save the plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot overall comparisons
    plot_metric_comparison(results, output_dir / "metric_comparison.png")
    plot_token_vs_qa(results, output_dir / "token_vs_qa.png")
    
    # Plot distributions of metrics
    metrics_to_plot = [
        "qa_score", "tokens_used", "token_efficiency", "keep_ratio", "successful_keep_ratio", "tokens_kept_ratio"
    ]
    
    for metric in metrics_to_plot:
        plot_metric_distribution(
            results, metric, output_dir / f"{metric}_distribution.png"
        )
    
    # Plot per-dataset comparisons
    if "per_dataset_improvements" in results:
        plot_per_dataset_comparison(results, output_dir / "per_dataset_comparison.png")
        
        # Save per-dataset details to CSV
        per_dataset_df = create_per_dataset_dataframe(results)
        per_dataset_df.to_csv(output_dir / "per_dataset_metrics.csv", index=False)


def plot_metric_distribution(results: Dict[str, Any], metric: str, output_path: str):
    """
    Plot distribution of a metric for both SPARC and baseline.
    
    Args:
        results: The evaluation results
        metric: Metric to plot
        output_path: Path to save the plot
    """
    if metric not in results["sparc"]["metrics"]:
        return  # Skip if metric doesn't exist
    
    # Set up figure
    plt.figure(figsize=(10, 6))
    
    # Create dataframe
    data = {
        "System": ["SPARC"] * len(results["sparc"]["metrics"][metric]) + 
                 ["Baseline"] * len(results["baseline"]["metrics"][metric]),
        "Value": results["sparc"]["metrics"][metric] + results["baseline"]["metrics"][metric]
    }
    if "dataset" in results["sparc"]["metrics"]:
        data["Dataset"] = results["sparc"]["metrics"]["dataset"] + results["baseline"]["metrics"]["dataset"]
    
    df = pd.DataFrame(data)
    
    # Get proper title
    metric_titles = {
        "qa_score": "QA Score",
        "em_score": "Exact Match Score",
        "f1_score": "F1 Score",
        "tokens_used": "Tokens Used",
        "token_efficiency": "Token Efficiency",
        "keep_ratio": "Keep Ratio",
        "successful_keep_ratio": "Successful Keep Ratio",
        "tokens_kept_ratio": "Tokens Kept Ratio"
    }
    title = metric_titles.get(metric, metric.replace("_", " ").title())
    
    # Create plot based on available data
    if "Dataset" in data:
        g = sns.catplot(
            data=df, 
            kind="violin", 
            x="System", 
            y="Value", 
            hue="Dataset", 
            split=True, 
            inner="quart", 
            palette="Set2"
        )
        g.fig.suptitle(f"{title} Distribution by Dataset")
    else:
        g = sns.violinplot(x="System", y="Value", data=df)
        plt.title(f"{title} Distribution")
    
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_metric_comparison(results: Dict[str, Any], output_path: str):
    """
    Plot comparison of all metrics between SPARC and baseline.
    
    Args:
        results: The evaluation results
        output_path: Path to save the plot
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get metrics with improvements
    metrics = list(results["improvements"].keys())
    metric_names = {
        "qa_score": "QA Score",
        "em_score": "Exact Match",
        "f1_score": "F1 Score",
        "tokens_used": "Token Savings",
        "token_efficiency": "Token Efficiency"
    }
    
    # Get improvement values
    values = [results["improvements"][m] for m in metrics]
    labels = [metric_names.get(m, m) for m in metrics]
    
    # Create color-coded bars (green for positive, red for negative)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    
    # Plot
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("SPARC Improvements Over Baseline")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_token_vs_qa(results: Dict[str, Any], output_path: str):
    """
    Create a scatter plot of tokens used vs QA score for both SPARC and baseline.
    
    Args:
        results: The evaluation results
        output_path: Path to save the plot
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    sparc_tokens = [ep["tokens_used"] for ep in results["sparc"]["episodes"]]
    sparc_qa = [ep["qa_score"] for ep in results["sparc"]["episodes"]]
    baseline_tokens = [ep["tokens_used"] for ep in results["baseline"]["episodes"]]
    baseline_qa = [ep["qa_score"] for ep in results["baseline"]["episodes"]]
    
    # Add dataset information if available
    if "dataset" in results["sparc"]["episodes"][0]:
        sparc_datasets = [ep["dataset"] for ep in results["sparc"]["episodes"]]
        baseline_datasets = [ep["dataset"] for ep in results["baseline"]["episodes"]]
        
        # Create dataframe with dataset info
        df = pd.DataFrame({
            "Tokens": sparc_tokens + baseline_tokens,
            "QA Score": sparc_qa + baseline_qa,
            "System": ["SPARC"] * len(sparc_tokens) + ["Baseline"] * len(baseline_tokens),
            "Dataset": sparc_datasets + baseline_datasets
        })
        
        # Create scatter plot with dataset colors
        sns.scatterplot(
            data=df, 
            x="Tokens", 
            y="QA Score", 
            hue="Dataset", 
            style="System", 
            palette="Set2", 
            s=80,
            alpha=0.7
        )
    else:
        # Create simple dataframe without dataset info
        df = pd.DataFrame({
            "Tokens": sparc_tokens + baseline_tokens,
            "QA Score": sparc_qa + baseline_qa,
            "System": ["SPARC"] * len(sparc_tokens) + ["Baseline"] * len(baseline_tokens)
        })
        
        # Create scatter plot without dataset colors
        sns.scatterplot(
            data=df, 
            x="Tokens", 
            y="QA Score", 
            hue="System", 
            s=80,
            alpha=0.7
        )
    
    # Add means as larger markers
    sparc_mean_tokens = results["sparc"]["tokens_used_mean"]
    sparc_mean_qa = results["sparc"]["qa_score_mean"]
    baseline_mean_tokens = results["baseline"]["tokens_used_mean"]
    baseline_mean_qa = results["baseline"]["qa_score_mean"]
    
    plt.scatter([sparc_mean_tokens], [sparc_mean_qa], s=200, color="blue", label="SPARC Mean", marker="*", edgecolor="black")
    plt.scatter([baseline_mean_tokens], [baseline_mean_qa], s=200, color="orange", label="Baseline Mean", marker="*", edgecolor="black")
    
    # Draw line connecting means
    plt.plot([sparc_mean_tokens, baseline_mean_tokens], [sparc_mean_qa, baseline_mean_qa], 'k--', alpha=0.5)
    
    # Annotate improvement percentages
    plt.annotate(
        f"{results['improvements']['qa_score']:.1f}% QA\n{results['improvements']['tokens_used']:.1f}% Tokens", 
        xy=((sparc_mean_tokens + baseline_mean_tokens)/2, (sparc_mean_qa + baseline_mean_qa)/2),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5")
    )
    
    # Set labels and title
    plt.xlabel("Tokens Used")
    plt.ylabel("QA Score")
    plt.title("Token Usage vs QA Performance")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def log_to_wandb(results: Dict[str, Any], cfg: DictConfig, model_path: str):
    """
    Log evaluation results to Weights & Biases.
    
    Args:
        results: Evaluation results
        cfg: Configuration dictionary
        model_path: Path to the model
    """
    try:
        import wandb
        
        # Initialize wandb
        wandb.init(
            project="sparc-eval",
            name=f"{Path(model_path).stem}-eval",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        
        # Log scalar metrics (overall)
        for system in ["sparc", "baseline"]:
            for metric in ["qa_score", "em_score", "f1_score", "tokens_used", "token_efficiency", "keep_ratio"]:
                if f"{metric}_mean" in results[system]:
                    wandb.log({f"{system}/{metric}": results[system][f"{metric}_mean"]})
        
        # Log improvements
        for metric, value in results["improvements"].items():
            wandb.log({f"improvement/{metric}": value})
        
        # Log per-dataset metrics if available
        if "per_dataset_improvements" in results:
            for dataset, metrics in results["per_dataset_improvements"].items():
                for metric, value in metrics.items():
                    wandb.log({f"dataset/{dataset}/{metric}": value})
        
        # Create and log comparison plots
        for plot_path in ["metric_comparison.png", "token_vs_qa.png", "per_dataset_comparison.png"]:
            if Path(plot_path).exists():
                wandb.log({f"plots/{Path(plot_path).stem}": wandb.Image(plot_path)})
        
        # Finish wandb session
        wandb.finish()
    except ImportError:
        print("Weights & Biases (wandb) not installed. Skipping logging.")
    except Exception as e:
        print(f"Error logging to wandb: {e}")
    
    # Log plots
    # These would be created by plot_results() and saved to output_dir
    
    # Finish run
    wandb.finish()


def prune_for_json(obj):
    """Recursively convert defaultdicts and remove unserializable/circular objects."""
    if isinstance(obj, dict):
        return {k: prune_for_json(v) for k, v in obj.items() if not callable(v)}
    elif isinstance(obj, list):
        return [prune_for_json(v) for v in obj]
    elif isinstance(obj, defaultdict):
        return prune_for_json(dict(obj))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # For other types (e.g., numpy, torch), try to convert, else skip
        try:
            return str(obj)
        except Exception:
            return None


def main():
    """
    Main function to run the evaluation script.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = None
    
    # Run evaluation
    results = evaluate_model(
        model_path=model_path,
        cfg=cfg,
        num_episodes=args.num_episodes,
        datasets=args.datasets,  # Use datasets list instead of single dataset
        per_dataset_episodes=args.per_dataset_episodes,
        split=args.split,
        seed=args.seed
    )
    
    # Print results to console
    print_results(results)
    
    # Save results
    if args.output_dir is not None:
        # Save results as JSON for later analysis
        results_file = Path(args.output_dir) / "results.json"
        with open(results_file, "w") as f:
            # Convert defaultdict to regular dict for serialization
            serializable_results = prune_for_json(results)
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {results_file}")
        
        # Create visualizations
        plot_results(results, args.output_dir)
        print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()