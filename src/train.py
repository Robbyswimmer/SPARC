# src/train.py
import hydra, wandb, torch, gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from transformers import AutoTokenizer
import itertools
from typing import Optional

from envs.streaming_qagym import StreamingQAGym
from agents.ppo_agent import TokenEmbedExtractor, WandbCallback
from agents.entropy_schedule import EntropyScheduleCallback
from agents.global_step_callback import GlobalStepCallback
from data.narrativeqa import load_narrativeqa
from data.hotpotqa    import load_hotpotqa
from curricula.length_schedule import LengthScheduleWrapper


class CurriculumStepCallback(BaseCallback):
    """
    Callback that advances curriculum step counter based on actual training steps.
    
    This ensures the curriculum progresses properly in RL training, where the environment
    is only reset at episode completion rather than every training step.
    
    Args:
        curriculum_wrapper: The LengthScheduleWrapper instance to step forward
        frequency: How often to advance the curriculum counter (in timesteps)
        verbose: Verbosity level (0 for no output, 1 for info output, 2 for debug output)
    """
    def __init__(self, curriculum_wrapper: LengthScheduleWrapper, frequency: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_wrapper = curriculum_wrapper
        self.frequency = frequency
        
    def _on_step(self) -> bool:
        # Advance the curriculum every [frequency] steps
        if self.n_calls % self.frequency == 0:
            self.curriculum_wrapper.step(1)
            # Log current curriculum level occasionally
            if self.n_calls % 1000 == 0:  # Report less frequently
                current_max = self.curriculum_wrapper.get_current_max_chunks()
                print(f"Step {self.n_calls}: Curriculum max chunks = {current_max}")
        return True

DATASETS = {
    "narrativeqa": load_narrativeqa,
    "hotpotqa"   : load_hotpotqa,
    # add more loaders here
}

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # 1) Build joint iterable over chosen datasets
    loaders = [DATASETS[name](tokenizer=tokenizer,split=cfg.data.split) for name in cfg.data.datasets]
    mixed_loader = itertools.cycle(itertools.chain(*loaders))  # simple round‑robin

    # 2) Create curriculum wrapper if requested
    curriculum_wrapper = None
    if cfg.curriculum.enabled:
        curriculum_wrapper = LengthScheduleWrapper(
            mixed_loader,
            initial_max_chunks=cfg.curriculum.start_chunks,
            final_max_chunks=cfg.curriculum.max_chunks, 
            total_schedule_steps=cfg.curriculum.growth_steps,
        )
        mixed_loader = curriculum_wrapper

    # 3) Factory to create fresh envs that consume the loader
    def make_env(seed=None):
        def _thunk():
            return StreamingQAGym(
                chunk_size   = cfg.env.chunk_size,
                max_window   = cfg.env.max_window,
                data_iter    = mixed_loader,   # pass iterator
                seed         = seed,
            )
        return _thunk

    vec_env = DummyVecEnv([make_env(seed=i) for i in range(cfg.train.n_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class  = TokenEmbedExtractor,
        features_extractor_kwargs = dict(embed_dim=cfg.model.embed_dim,
                                         vocab_size=cfg.model.vocab_size),
        net_arch                  = [dict(pi=[128, 64], vf=[128, 64])],
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate = cfg.train.lr,
        n_steps       = cfg.train.n_steps,
        batch_size    = cfg.train.batch_size,
        gamma         = cfg.train.gamma,
        gae_lambda    = cfg.train.gae_lambda,
        clip_range    = cfg.train.clip_range,
        ent_coef      = 0.05,  # Start with high entropy for exploration
        policy_kwargs = policy_kwargs,
        tensorboard_log = "./runs/tb",
        device          = "auto",
    )

    run_name = f"{cfg.exp.name}-{wandb.util.generate_id()}"
    # Setup callbacks
    callbacks = [WandbCallback(project=cfg.wandb.project, run_name=run_name)]
    
    # Add curriculum callback if curriculum is enabled
    if cfg.curriculum.enabled and curriculum_wrapper is not None:
        curriculum_callback = CurriculumStepCallback(
            curriculum_wrapper=curriculum_wrapper,
            frequency=1  # Advance every step
        )
        callbacks.append(curriculum_callback)
        print("Using curriculum progression callback")
    
    # Add entropy schedule callback to linearly decay entropy coefficient
    entropy_callback = EntropyScheduleCallback(
        start_coef=0.05,
        end_coef=0.0,
        decay_fraction=0.5,  # Decay over first half of training
        total_timesteps=cfg.train.total_steps,
        verbose=1
    )
    callbacks.append(entropy_callback)
    print("Using entropy scheduling to control exploration")
    
    # Add global step callback for token reward annealing
    global_step_callback = GlobalStepCallback(verbose=1)
    callbacks.append(global_step_callback)
    print("Using global step tracking for token reward annealing")
    
    model.learn(
        total_timesteps = cfg.train.total_steps,
        callback        = callbacks
    )

    model.save(f"checkpoints/{run_name}.zip")

if __name__ == "__main__":
    main()
