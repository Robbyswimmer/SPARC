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
    """Embed token‑ID observations and pool into a vector."""
    def __init__(self, observation_space: gym.Space, embed_dim: int = 128, vocab_size: int = 128256):
        super().__init__(observation_space, features_dim=embed_dim)
        chunk_size = observation_space.shape[0]
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.chunk_size = chunk_size

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.long()                       # (batch, chunk_size)
        x = self.embed(obs)                    # (batch, chunk_size, embed_dim)
        x = x.mean(dim=1)                      # (batch, embed_dim)
        return x

# -----------------------------------------------------------------------------
class WandbCallback(BaseCallback):
    """Log raw env info dicts to Weights & Biases."""
    def __init__(self, project: str, run_name: str, ema_alpha: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.run_name = run_name
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
        self._wandb = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.model.get_parameters(),
            reinit=True
        )

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
                else:
                    print(f"DEBUG (Env {i}): No QA metrics found in info dict")
        return True

    def _on_training_end(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

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
