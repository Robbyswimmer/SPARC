#!/usr/bin/env python3
# requires: conda activate streamqa
"""
Random agent demo for StreamingQAGym.

This script demonstrates a random policy agent interacting with the
StreamingQAGym environment. It visualizes the agent's decisions (KEEP/DROP)
and tracks statistics about token usage and rewards.
"""
import gymnasium as gym
import sys
import os
import time
import random
from typing import List, Dict, Any, Tuple

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.streaming_qagym import StreamingQAGym, tokenizer
import numpy as np

# ANSI color codes for pretty printing
COLORS = {
    'green': '\033[92m',
    'red': '\033[91m',
    'blue': '\033[94m',
    'yellow': '\033[93m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    'end': '\033[0m'
}

def colorize(text: str, color: str) -> str:
    """Add color to terminal text."""
    return f"{COLORS.get(color, '')}{text}{COLORS['end']}"

def print_step_info(step: int, action: int, reward: float, info: Dict[str, Any]) -> None:
    """Print information about the current step."""
    action_str = colorize("KEEP", "green") if action == 1 else colorize("DROP", "red")
    print(f"Step {step:2d} | Action: {action_str} | Reward: {reward:.4f}")
    
    # Print additional info if available
    if 'error' in info:
        print(f"  {colorize('Error:', 'red')} {info['error']}")
    if 'question' in info and info['question']:
        print(f"  {colorize('Question presented', 'blue')}")

def print_episode_summary(steps: int, actions: List[int], rewards: List[float], 
                       final_info: Dict[str, Any], total_time: float) -> None:
    """Print a summary of the episode."""
    keep_count = sum(1 for a in actions if a == 1)
    drop_count = sum(1 for a in actions if a == 0)
    keep_ratio = keep_count / len(actions) if actions else 0
    
    print("\n" + "=" * 50)
    print(colorize("EPISODE SUMMARY", "bold"))
    print("=" * 50)
    print(f"Steps completed: {steps}")
    print(f"Time elapsed: {total_time:.2f} seconds")
    print(f"Final reward: {sum(rewards):.4f}")
    print(f"Action distribution: {keep_count} KEEP ({keep_ratio:.1%}), {drop_count} DROP ({1-keep_ratio:.1%})")
    
    if 'tokens_used' in final_info:
        print(f"Tokens used: {final_info['tokens_used']} / {2048} ({final_info['tokens_used']/2048:.1%})")
    
    if 'qa_score' in final_info:
        print(f"QA Score: {final_info['qa_score']:.4f}")
    
    if 'model_answer' in final_info and 'gold_answer' in final_info:
        print("\n" + colorize("QUESTION ANSWERING RESULTS", "bold"))
        print(f"Gold answer: {colorize(final_info['gold_answer'], 'yellow')}")
        print(f"Model answer: {colorize(final_info['model_answer'], 'blue')}")

def run_random_agent(env_config: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
    """Run a random agent in the StreamingQAGym environment."""
    env_config = env_config or {}
    env = StreamingQAGym(**env_config)
    
    print(colorize("\nInitializing StreamingQAGym environment...", "bold"))
    obs, info = env.reset()
    print(colorize("Environment ready. Starting episode with random policy.", "bold"))
    
    done = False
    step_count = 0
    actions = []
    rewards = []
    start_time = time.time()
    
    try:
        while not done:
            # Random policy
            action = env.action_space.sample()
            actions.append(action)
            
            # Take step in environment
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
            
            # Print step information
            print_step_info(step_count, action, reward, info)
            step_count += 1
            
            # Small delay for readability
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print(colorize("\nEpisode interrupted by user.", "red"))
    
    except Exception as e:
        print(colorize(f"\nError during episode: {e}", "red"))
    
    finally:
        total_time = time.time() - start_time
        print_episode_summary(step_count, actions, rewards, info, total_time)
        return sum(rewards), info

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Configure environment
    config = {
        "max_window": 2048,  # Maximum context window size
        "chunk_size": 256,   # Size of each document chunk
        "seed": 42           # Environment seed
    }
    
    # Run the demo
    total_reward, final_info = run_random_agent(config)
