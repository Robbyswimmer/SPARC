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
import argparse
import datetime as dt
from typing import Dict, Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import wandb
import numpy as np
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
    def __init__(self, project: str, run_name: str, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.run_name = run_name
        self._wandb = None

    def _on_training_start(self) -> None:
        self._wandb = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.model.get_parameters(),
            reinit=True
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # Log any info dict containing qa_score or episode_reward
        for info in infos:
            if any(k in info for k in ("qa_score", "episode_reward")):
                # flatten numpy types
                clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in info.items()}
                self._wandb.log(clean, step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "total_timesteps": 200_000,
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
