"""PPO training harness for SPARC using Stable‑Baselines3 (>=2.3).

Key features
------------
* Vectorised StreamingQAGym via SB3 `VecEnv` wrappers (no VecMonitor to preserve env infos).
* TokenEmbedExtractor for embedding token‑ID observations.
* WandB callback logs raw step infos (including qa_score, episode_reward, tokens_used).
* Configurable via DEFAULT_CONFIG.

Usage
-----
>>> python -m src.agents.ppo_agent --total_timesteps 200_000

Requirements: stable-baselines3==2.3, wandb, gymnasium>=0.29, torch>=2.1
"""
from __future__ import annotations

import os
from typing import Dict, Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import wandb
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.streaming_qagym import StreamingQAGym

# -----------------------------------------------------------------------------
class TokenEmbedExtractor(BaseFeaturesExtractor):
    """Embed token‑ID chunk and question, pool and concatenate for policy input."""
    def __init__(self, observation_space: gym.Space, embed_dim: int = 128, vocab_size: int = 128256):
        # Output dim is 2*embed_dim: [chunk_emb, question_emb]
        super().__init__(observation_space, features_dim=2 * embed_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.chunk_size = observation_space.spaces["chunk"].shape[0]
        self.question_size = observation_space.spaces["question"].shape[0]

    def forward(self, obs: dict) -> th.Tensor:
        # obs is a dict: {"chunk": Tensor, "question": Tensor}
        chunk = obs["chunk"].long()        # (batch, chunk_size)
        question = obs["question"].long()  # (batch, question_size)
        chunk_emb = self.embed(chunk).mean(dim=1)        # (batch, embed_dim)
        question_emb = self.embed(question).mean(dim=1)  # (batch, embed_dim)
        x = th.cat([chunk_emb, question_emb], dim=-1)    # (batch, 2*embed_dim)
        return x


# -----------------------------------------------------------------------------
class WandbCallback(BaseCallback):
    """Log raw env info dicts to Weights & Biases."""
    def __init__(self, project: str, run_name: str, config=None, log_hindsight_bonus_steps: int = 0, ema_alpha: float = 0.05, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.project = project
        self.run_name = run_name
        self.config_to_log = config  # Store the configuration to log
        self.log_hindsight_bonus_steps = log_hindsight_bonus_steps
        self._wandb = None
        
        # EMA tracking for smoothed metrics
        self.ema_alpha = ema_alpha  # Lower alpha = more smoothing (0.05 is quite smooth)
        self.loss_metrics_last_logged = 0  # Track the last n_updates when loss was logged
        self.smoothed_metrics = {
            "qa_score_ema": None,
            "f1_ema": None,
            "em_ema": None,
            "episode_reward_ema": None,
            "tokens_used_ema": None,
            "keep_count_ema": None,
            "drop_count_ema": None,
            "keep_ratio_ema": None
        }

    def _on_training_start(self) -> None:
        # Initialize the wandb run without complex config objects that might cause serialization issues
        self._wandb = wandb.init(
            project=self.project,
            name=self.run_name,
            # Only include simple model parameters to avoid serialization issues
            config={
                "learning_rate": self.model.learning_rate,
                "n_steps": self.model.n_steps,
                "batch_size": self.model.batch_size,
                "gamma": self.model.gamma,
                "gae_lambda": self.model.gae_lambda,
                "clip_range": self.model.clip_range,
                "n_envs": self.model.n_envs
            },
            reinit=True
        )
        
        # Log additional config info separately if provided
        if self.config_to_log is not None:
            try:
                # Extract only basic config attributes for logging
                if hasattr(self.config_to_log, 'env'):
                    env_config = {}
                    if hasattr(self.config_to_log.env, 'reward'):
                        reward_config = {}
                        for key in ['alpha', 'beta_keep', 'gamma_step', 'kappa', 'use_hindsight']:
                            if hasattr(self.config_to_log.env.reward, key):
                                reward_config[key] = getattr(self.config_to_log.env.reward, key)
                        env_config['reward'] = reward_config
                    self._wandb.config.update({"env": env_config}, allow_val_change=True)
            except Exception as e:
                print(f"Warning: Could not fully log config to wandb: {e}")

    def _on_step(self) -> bool:
        # 1) grab everything SB3 has just recorded into its logger
        sb3_metrics: dict = self.model.logger.name_to_value

        # filter down to the 'train/' metrics you care about:
        train_metrics = {k: v for k, v in sb3_metrics.items() if k.startswith("train/")}
        # add a global step so WandB will put them on the x‑axis
        train_metrics["global_step"] = self.num_timesteps
        self._wandb.log(train_metrics)

        # 2) Log loss metrics if present
        # These are available during the update phase
        loss_vals = self.locals.get("loss_vals")
        if loss_vals is not None and self.model.n_updates > self.loss_metrics_last_logged:
            # Track that we've logged for this update step
            self.loss_metrics_last_logged = self.model.n_updates
            loss_log_data = {
                "loss/policy": float(loss_vals.get('pg_loss', 0)),
                "loss/value": float(loss_vals.get('value_loss', 0)),
                "loss/entropy": float(loss_vals.get('entropy', 0)),
                "loss/approx_kl": float(loss_vals.get('approx_kl', 0)),
                "loss/clip_fraction": float(loss_vals.get('clip_fraction', 0)),
                "loss/loss": float(loss_vals.get('loss', 0)),
                "global_step": self.num_timesteps,
                "n_updates": self.model.n_updates,
            }
            self._wandb.log(loss_log_data)

        # 3) Log episode metrics upon termination
        # Stable Baselines3 provides episode info in `infos` when `dones` is True
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if infos is None or dones is None:
            print("WandbCallback: infos or dones is None, skipping episode logging.")
            return True # Should not happen, but safety check

        # Iterate through each environment's info and done status
        for i in range(self.training_env.num_envs):
            if dones[i]: # Check if THIS environment terminated
                print(f"--- Env {i} Episode Terminated (Step {self.num_timesteps}) ---") # DEBUG
                info = infos[i] # Get the info dict for this specific env
                episode_info = info.get("episode", {}) # Access episode info
                
                # Get metrics directly from info dict if they exist
                qa_score = info.get("qa_score", None)
                em = info.get("exact_match", None)
                f1 = info.get("f1", None)
                tokens_used = info.get("tokens_used", None)
                keep_count = info.get("keep_count", None)  # Get keep_count
                drop_count = info.get("drop_count", None)  # Get drop_count
                
                # Initialize keep_ratio here to avoid UnboundLocalError
                keep_ratio = None
                if keep_count is not None and drop_count is not None:
                    total_decisions = keep_count + drop_count
                    if total_decisions > 0:
                        keep_ratio = keep_count / total_decisions

                if qa_score is not None:
                    # Update EMAs
                    episode_reward = episode_info.get("r", 0.0)
                    
                    # Update each EMA
                    for metric_name, current_value in [
                        ("qa_score_ema", qa_score),
                        ("f1_ema", f1),
                        ("em_ema", em),
                        ("episode_reward_ema", episode_reward),
                        ("tokens_used_ema", tokens_used),
                        ("keep_count_ema", keep_count),
                        ("drop_count_ema", drop_count),
                        ("keep_ratio_ema", keep_ratio)
                    ]:
                        if current_value is not None:
                            if self.smoothed_metrics[metric_name] is None:
                                # Initialize with first value
                                self.smoothed_metrics[metric_name] = current_value
                            else:
                                # EMA update: new_ema = alpha * current + (1-alpha) * old_ema
                                self.smoothed_metrics[metric_name] = (
                                    self.ema_alpha * current_value + 
                                    (1 - self.ema_alpha) * self.smoothed_metrics[metric_name]
                                )
                    
                    # Log both raw and smoothed metrics
                    log_data = {
                        "episode_reward": episode_reward,
                        "episode_length": episode_info.get("l", 0), 
                        "em": em,
                        "f1": f1,
                        "qa_score": qa_score,
                        "tokens_used": tokens_used,
                        "keep_count": keep_count,  # Log keep_count
                        "drop_count": drop_count,  # Log drop_count
                        "keep_ratio": keep_ratio,  # Log keep ratio
                        "global_step": self.num_timesteps,
                    }
                    
                    # Add smoothed metrics to log data
                    for metric_name, smoothed_value in self.smoothed_metrics.items():
                        if smoothed_value is not None:
                            log_data[metric_name] = smoothed_value
                    
                    log_data = {k: v for k, v in log_data.items() if v is not None}
                    self._wandb.log(log_data)
                    
                    # Log hindsight bonus for first N steps
                    hindsight_bonus_pkc = info.get("hindsight_bonus_pkc", None)
                    if hindsight_bonus_pkc is not None and self.num_timesteps < self.log_hindsight_bonus_steps:
                        self._wandb.log({"debug/hindsight_bonus_pkc": hindsight_bonus_pkc})
                else:
                    print(f"DEBUG (Env {i}): No QA metrics found in info dict")
        return True

    def _on_training_end(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

# -----------------------------------------------------------------------------
class HindsightRewardCallback(BaseCallback):
    """Callback to apply hindsight credits during training.
    
    Since we can't easily access the episode info dicts from the DictRolloutBuffer,
    we take a simpler approach: monitor episodes and track hindsight info when episodes
    terminate, then apply hindsight credits at the next rollout_end.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Store hindsight credits to be applied at next rollout_end
        self.pending_credits = {}  # {env_idx: [(step_idx, reward_value), ...]}
        self.episode_step_count = {}  # Track steps in current episode for each env
        self.buffer_offset = 0  # Tracks step count across rollouts
        
    def _on_step(self) -> bool:
        """Track episode steps and monitor for terminations with hindsight rewards."""
        # Get latest info dicts
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        
        if infos is not None and dones is not None:
            # Increment step counters for all envs
            for env_idx in range(len(infos)):
                if env_idx not in self.episode_step_count:
                    self.episode_step_count[env_idx] = 0
                self.episode_step_count[env_idx] += 1
                
                # Check for episode termination
                if dones[env_idx]:
                    info = infos[env_idx]
                    hindsight_rewards = info.get("hindsight_adjusted_rewards")
                    
                    if hindsight_rewards and self.episode_step_count[env_idx] > 0:
                        if self.verbose > 0:
                            print(f"-- Env {env_idx} Episode Terminated (Step {self.num_timesteps}) ---")
                            
                        # Calculate the global step indices for this episode
                        episode_length = min(self.episode_step_count[env_idx], len(hindsight_rewards))
                        
                        # Store hindsight rewards for this env to apply at rollout_end
                        self.pending_credits[env_idx] = []
                        
                        # Calculate offset from current buffer position to start of this episode
                        current_position_in_buffer = self.num_timesteps % self.model.n_steps
                        episode_start_offset = episode_length - 1
                        
                        # Store each step's reward and its position
                        for i in range(episode_length):
                            # This step's buffer position (counting backwards from current position)
                            buffer_pos = (current_position_in_buffer - i) % self.model.n_steps
                            # This step's reward index in the episode history
                            reward_idx = episode_length - 1 - i
                            
                            # Only store if indices are valid
                            if reward_idx < len(hindsight_rewards):
                                self.pending_credits[env_idx].append((buffer_pos, hindsight_rewards[reward_idx]))
                    
                    # Reset episode step counter for this env
                    self.episode_step_count[env_idx] = 0
        
        return True

    def _on_rollout_end(self) -> None:
        """Apply any pending hindsight credits to the rollout buffer."""
        buffer = self.model.rollout_buffer
        buffer_size = buffer.buffer_size

        if self.verbose > 0:
            print(f"[HindsightRewardCallback] Processing {len(self.pending_credits)} envs with credits, buffer size {buffer_size}.")

        # Apply pending credits to buffer
        credits_applied = 0
        for env_idx, credit_list in list(self.pending_credits.items()):
            for buffer_pos, reward_value in credit_list:
                # Apply the credit if the buffer position is within the current buffer
                if 0 <= buffer_pos < buffer_size:
                    buffer.rewards[buffer_pos, env_idx] = reward_value
                    credits_applied += 1
            
            # Clear pending credits for this env
            del self.pending_credits[env_idx]
            
        if self.verbose > 0 and credits_applied > 0:
            print(f"[HindsightRewardCallback] Applied {credits_applied} hindsight credits to buffer.")

        # Update buffer offset for next rollout
        self.buffer_offset += buffer_size

# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "total_timesteps": 100_000,
    "learning_rate": 3e-5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_steps": 1024,
    "batch_size": 256,
    "n_envs": 1,       # single env to preserve infos
}

# -----------------------------------------------------------------------------
def make_env(seed: int | None = None):
    def thunk():
        env = StreamingQAGym(seed=seed)
        return env
    return thunk

# -----------------------------------------------------------------------------
def train(config: Dict[str, Any] | None = None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    # create vectorized env (no VecMonitor)
    env_fns = [make_env(seed=i) for i in range(cfg["n_envs"])]
    vec_env = DummyVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=TokenEmbedExtractor,
        features_extractor_kwargs=dict(embed_dim=128, vocab_size=128256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./runs/tb",
        device="auto",
    )

    run_name = f"sparc-ppo-{dt.datetime.now():%Y%m%d-%H%M%S}"
    callback = WandbCallback(project="SPARC", run_name=run_name)

    model.learn(total_timesteps=cfg["total_timesteps"], callback=callback)

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"ppo_sparc_{cfg['total_timesteps']}.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_CONFIG["total_timesteps"])
    args = parser.parse_args()
    train({"total_timesteps": args.total_timesteps})
