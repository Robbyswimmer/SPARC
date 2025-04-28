"""
Evaluation callback for SPARC agent.

This module provides a callback for periodically evaluating the agent's 
performance on a fixed validation set during training.
"""
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import wandb
import os
from typing import Callable, Any
from envs.streaming_qagym import StreamingQAGym

class ValidationCallback(BaseCallback):
    """
    Callback for evaluating the agent on a separate validation set.
    
    This callback periodically evaluates the model on a validation environment
    and logs metrics to track performance independent of the training data.
    It also saves the best model found so far based on mean reward.
    
    Args:
        eval_freq: Evaluate every `eval_freq` timesteps
        eval_episodes: Number of episodes to evaluate on
        data_loader_fn: Function that returns a validation data iterator
        save_path: Path to save the best model checkpoint
        name_prefix: Prefix for the saved model file
        n_eval_envs: Number of parallel environments for evaluation
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
        deterministic: Whether to use deterministic actions during evaluation
    """
    
    def __init__(
        self,
        eval_freq: int = 10000,
        eval_episodes: int = 20,
        data_loader_fn: Callable[[], Any] = None,
        n_eval_envs: int = 1,
        save_path: str = "./checkpoints/",
        name_prefix: str = "best_model",
        verbose: int = 1,
        deterministic: bool = True,
    ):
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.data_loader_fn = data_loader_fn
        self.n_eval_envs = n_eval_envs
        self.deterministic = deterministic
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        # Initialize metrics storage
        self.best_mean_reward = -np.inf
        self.last_metrics = {}
        
        # Validation environment will be created in _on_training_start
        self.eval_env = None
    
    def _on_training_start(self) -> None:
        """
        Create validation environment when training starts.
        """
        if self.data_loader_fn is None:
            raise ValueError("Validation data loader function must be provided")
        
        # Create validation environment with dedicated validation data
        def make_eval_env():
            return StreamingQAGym(
                data_loader_fn=self.data_loader_fn,  # Pass the function itself
                # Forward other params from training env as needed
                max_window=self.training_env.envs[0].max_window,
                chunk_size=self.training_env.envs[0].chunk_size,
                # No curriculum for validation to keep consistent measurement
            )
        
        # Create vectorized validation environment
        self.eval_env = DummyVecEnv([make_eval_env for _ in range(self.n_eval_envs)])
        
        if self.verbose > 0:
            print(f"Validation environment created with {self.n_eval_envs} parallel envs")
    
    def _on_step(self) -> bool:
        """
        Evaluate the agent on validation set at regular intervals.
        
        Returns:
            True to continue training
        """
        # Debug: Print timestep info every 10 steps
        if self.verbose > 0 and self.n_calls % 10 == 0:
            print(f"Debug: ValidationCallback - n_calls={self.n_calls}, num_timesteps={self.num_timesteps}, eval_freq={self.eval_freq}")
            print(f"Debug: Next validation at timestep {(self.num_timesteps // self.eval_freq + 1) * self.eval_freq if self.num_timesteps > 0 else self.eval_freq}")
        
        # Skip if not at evaluation frequency or no eval env
        if self.eval_env is None or self.num_timesteps % self.eval_freq != 0:
            return True
        
        if self.verbose > 0:
            print(f"\nRunning validation at timestep {self.num_timesteps}...")
        
        # Initialize metrics collectors
        episode_rewards = []
        episode_lengths = []
        qa_scores = []
        em_scores = []
        f1_scores = []
        max_em_scores = []  # Track max-over-references EM
        max_f1_scores = []  # Track max-over-references F1
        tokens_used = []
        
        # Run evaluation episodes manually to collect detailed metrics
        for episode in range(self.eval_episodes):
            # Reset the environment (handle both old and new gym API)
            reset_result = self.eval_env.reset()
            # Handle different return types from reset() - either just obs or (obs, info)
            if isinstance(reset_result, tuple) and len(reset_result) >= 1:
                obs = reset_result[0]  # New API: returns (obs, info)
            else:
                obs = reset_result      # Old API: returns just obs
            
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_info = {}
            
            # Step through episode
            while not done:
                # Use the model to predict actions
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                
                # Take action in environment (handle both old and new gym API)
                step_result = self.eval_env.step(action)
                
                # Handle different return types from step()
                if len(step_result) == 5:  # New API: returns (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    # Check if episode is done (any environment in the vectorized env)
                    done = terminated.any() or truncated.any()
                else:  # Old API: returns (obs, reward, done, info)
                    obs, reward, done_old, info = step_result
                    # done is already a boolean in the old API
                    done = done_old if isinstance(done_old, bool) else done_old.any()
                
                # Update metrics
                episode_reward += reward[0]  # Take first env's reward
                episode_length += 1
                
                # If done, capture final metrics
                if done:
                    episode_info = info[0]  # Take first env's info
            
            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Extract QA metrics if available
            if "qa_score" in episode_info:
                qa_scores.append(episode_info.get("qa_score", 0.0))
                em_scores.append(episode_info.get("exact_match", 0.0))
                f1_scores.append(episode_info.get("f1", 0.0))
                # Get max-over-references metrics if available
                max_em_scores.append(episode_info.get("max_em", episode_info.get("exact_match", 0.0)))
                max_f1_scores.append(episode_info.get("max_f1", episode_info.get("f1", 0.0)))
                tokens_used.append(episode_info.get("tokens_used", 0))
        
        # Calculate summary statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        # Store for later reference
        self.last_metrics = {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/mean_length": mean_length,
            "global_step": self.num_timesteps,
        }
        
        # Add QA metrics if available
        if qa_scores:
            self.last_metrics.update({
                "eval/mean_qa_score": np.mean(qa_scores),
                "eval/mean_em": np.mean(em_scores),
                "eval/mean_f1": np.mean(f1_scores),
                "eval/mean_max_em": np.mean(max_em_scores),  # Log max-over-references EM
                "eval/mean_max_f1": np.mean(max_f1_scores),  # Log max-over-references F1
                "eval/mean_tokens_used": np.mean(tokens_used),
            })
        
        # Save best model
        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print(f"New best mean reward: {mean_reward:.3f} (old: {self.best_mean_reward:.3f}). Saving model...")
            self.best_mean_reward = mean_reward
            # Create save directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            save_filename = os.path.join(self.save_path, f"{self.name_prefix}.zip")
            self.model.save(save_filename)
            if self.verbose > 0:
                print(f"Model saved to {save_filename}")
        else:
            if self.verbose > 0:
                print(f"Mean reward {mean_reward:.3f} did not improve over best {self.best_mean_reward:.3f}.")
        
        # Log to wandb if available
        try:
            if wandb.run is not None:
                # Log scalar metrics
                wandb.log(self.last_metrics)
                
                # Log histograms for distributions
                if len(episode_rewards) > 1:  # Only create histograms if we have multiple episodes
                    # Log reward and episode length histograms
                    wandb.log({
                        "eval/reward_hist": wandb.Histogram(episode_rewards),
                        "eval/length_hist": wandb.Histogram(episode_lengths),
                        "global_step": self.num_timesteps,
                    })
                    
                    # Log QA metric histograms if available
                    if qa_scores:
                        wandb.log({
                            "eval/qa_score_hist": wandb.Histogram(qa_scores),
                            "eval/em_hist": wandb.Histogram(em_scores),
                            "eval/f1_hist": wandb.Histogram(f1_scores),
                            "eval/tokens_hist": wandb.Histogram(tokens_used),
                            "global_step": self.num_timesteps,
                        })
        except (ImportError, AttributeError):
            if self.verbose > 0:
                print("WandB not available for validation logging")
        
        # Print a summary
        if self.verbose > 0:
            print(f"Validation results ({self.eval_episodes} episodes):")
            print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            if qa_scores:
                print(f"  Mean QA score: {np.mean(qa_scores):.3f}")
                print(f"  Mean EM: {np.mean(em_scores):.3f}")
                print(f"  Mean F1: {np.mean(f1_scores):.3f}")
                print(f"  Mean tokens: {np.mean(tokens_used):.1f}")
        
        return True
